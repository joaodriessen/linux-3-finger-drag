//! The gesture state machine: the pure, I/O-free heart of the proxy.
//!
//! Everything time- and state-sensitive about classifying touches lives
//! here, decoupled from the devices themselves: the machine consumes raw
//! evdev frames (as plain `Ev` triples) plus an explicit `Instant`, and
//! emits explicit [`Output`] actions for the I/O shell (`mt_proxy`) to
//! carry out. No file descriptors, no wall clock, no side effects --
//! which means every historical regression this project has ever hit
//! (phantom right-clicks from staggered liftoffs, 4-finger swipes
//! misread as drags, stuck virtual buttons) is reproducible in a plain
//! `cargo test`, no fingers required. See the `tests` module at the
//! bottom: that suite is the project's institutional memory.
//!
//! Overview of the classification model (unchanged from the proven
//! design, now with the state isolated):
//!
//! * A "touch" spans from the first finger down to the last finger up,
//!   tracked as a whole -- never judged frame-by-frame, because real
//!   fingers land and lift asynchronously.
//! * A fresh touch is *buffered* (withheld from the compositor) until it
//!   is classified: a lone finger settles after a short `probe_delay`
//!   (so ordinary pointer motion never feels delayed), an ambiguous 2-3
//!   finger touch waits out `entry_debounce`, and reaching 4+ fingers
//!   settles it instantly (nothing with 4 fingers is ours).
//! * A touch that holds at exactly 3 fingers through the debounce window
//!   becomes a drag: buffered frames are discarded, the compositor never
//!   learns those fingers existed, and finger motion drives the virtual
//!   mouse instead.
//! * Once a drag starts it only ends when *every* finger lifts
//!   (hysteresis: staggered liftoff must not leak trailing 1-2 finger
//!   touches, which libinput would read as a right-click tap).
//! * A settled non-drag touch is relayed live, frame by frame, verbatim.

use std::time::{Duration, Instant};

use tracing::{debug, warn};

pub const EV_SYN: u16 = 0x00;
pub const EV_KEY: u16 = 0x01;
pub const EV_ABS: u16 = 0x03;
pub const SYN_REPORT: u16 = 0x00;
pub const SYN_DROPPED: u16 = 0x03;
pub const ABS_MT_SLOT: u16 = 0x2f;
pub const ABS_MT_TRACKING_ID: u16 = 0x39;
pub const ABS_MT_POSITION_X: u16 = 0x35;
pub const ABS_MT_POSITION_Y: u16 = 0x36;
pub const ABS_MT_TOUCH_MAJOR: u16 = 0x30;

/// Per-slot auxiliary MT axes forwarded alongside positions: contact
/// size/width/orientation/distance/pressure. libinput decides whether a
/// contact is a real touch partly from these (Apple pads have explicit
/// touch-size quirks) -- a clone touch that never reports a size is
/// filtered out entirely, killing gestures.
pub const MT_AUX_CODES: [u16; 7] = [0x30, 0x31, 0x32, 0x33, 0x34, 0x37, 0x3a];
const AUX_UNSEEN: i32 = i32::MIN;

/// Hard upper bound on tracked slots; the effective count comes from the
/// device's ABS_MT_SLOT range at construction.
pub const MAX_SLOTS: usize = 16;

/// Velocity band for the 4+ finger "flick", in PAD-LENGTHS PER SECOND
/// per axis: at or below the low bound the configured four_finger_scale
/// applies in full (calm, precise tracking); at or above the high bound
/// motion passes UNSCALED so a fast flick keeps enough travel to push
/// compositor gestures past their completion threshold instead of
/// bouncing back -- the macOS momentum feel. Smoothstepped in between.
/// Normalizing per axis matters: pads are wider than tall and vertical
/// flicks are biomechanically slower, so the same "fraction of the pad
/// per second" is the fair bar for both directions (and it's
/// resolution-independent across hardware).
const FLICK_LO_PADS_S: f64 = 0.9;
const FLICK_HI_PADS_S: f64 = 1.8;

/// Liftoff glide ("inertia"): when a flicked 4+ finger touch lifts, the
/// clone's fingers keep moving with decaying velocity before releasing,
/// so the gesture receives the travel the flick physically implied --
/// including whatever was lost to recognition latency and late-landing
/// fingers. Step cadence, per-step decay, minimum speed to keep
/// gliding, and a hard time cap.
const GLIDE_STEP: Duration = Duration::from_millis(10);
const GLIDE_DECAY: f64 = 0.82;
const GLIDE_MIN_PADS_S: f64 = 0.15;
const GLIDE_MAX: Duration = Duration::from_millis(240);

/// A raw evdev event stripped to the fields that matter. Mirrors
/// `input_event` minus the timestamp (the kernel re-stamps everything
/// written to uinput anyway).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ev {
    pub type_: u16,
    pub code: u16,
    pub value: i32,
}

impl Ev {
    pub const fn new(type_: u16, code: u16, value: i32) -> Self {
        Ev { type_, code, value }
    }
    pub const fn abs(code: u16, value: i32) -> Self {
        Ev::new(EV_ABS, code, value)
    }
    pub const fn syn() -> Self {
        Ev::new(EV_SYN, SYN_REPORT, 0)
    }
}

/// An effect the I/O shell must apply, in order.
#[derive(Clone, PartialEq, Debug)]
pub enum Output {
    /// Write these events to the synthetic touchpad clone.
    EmitSynth(Vec<Ev>),
    /// Press the virtual mouse's left button.
    MouseDown,
    /// Release the virtual mouse's left button.
    MouseUp,
    /// Drag motion: the reference finger's RAW delta in pad units. The
    /// I/O shell applies it to a synthetic finger on the clone, so drags
    /// ride the exact same libinput touchpad pipeline (acceleration
    /// curve, speed setting) as ordinary cursor movement -- identical
    /// feel by construction.
    MouseMove { dx: i32, dy: i32 },
}

/// The timing/scaling knobs the machine needs; derived from the user
/// config (and re-derivable on hot reload via [`GestureMachine::set_timing`]).
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub probe_delay: Duration,
    pub entry_debounce: Duration,
    /// 0 disables drag-lock entirely (the default). When > 0, lifting
    /// all fingers keeps the virtual button held for this long; a new
    /// touch that settles as a 3-finger drag resumes the same drag,
    /// while a touch that settles as anything else releases the button
    /// *before* its buffered events are relayed -- so ordinary pointer
    /// motion after a drag can never smear the held button around
    /// (the exact regression the first drag-lock attempt shipped).
    pub drag_end_delay: Duration,
    /// How long after a drag commits the button press is deferred when
    /// the fingers haven't moved yet. The press fires at the first
    /// actual drag motion or when this grace expires -- whichever comes
    /// first -- so a 4th finger landing *after* the entry window (a
    /// fast/sloppy 4-finger swipe, whose fingers stagger more the
    /// faster the hand comes down) can abort the misclassified drag
    /// without a phantom click ever having been sent.
    pub press_grace: Duration,
    /// Motion scale applied to touches of 4+ fingers as they are relayed
    /// to the compositor (anchored at each finger's touchdown position).
    /// Compositor gestures (KWin's 4-finger desktop-switch/overview) have
    /// no sensitivity setting of their own; because the compositor only
    /// sees what the proxy relays, this knob can slow them down without
    /// affecting cursor (1 finger), scroll (2) or drags (3). 1.0 = off.
    pub four_finger_scale: f64,
}

#[derive(Clone, Copy)]
struct Slot {
    tracking_id: i32,
    x: i32,
    y: i32,
    /// Values of the auxiliary MT axes (MT_AUX_CODES order); AUX_UNSEEN
    /// until the device first reports one. touch_major (index 0) doubles
    /// as the thumb/palm diagnostic (thumbs are much larger than
    /// fingertips).
    aux: [i32; MT_AUX_CODES.len()],
}

impl Slot {
    fn touch_major(&self) -> i32 {
        if self.aux[0] == AUX_UNSEEN {
            0
        } else {
            self.aux[0]
        }
    }
}

impl Default for Slot {
    fn default() -> Self {
        Slot {
            tracking_id: -1,
            x: 0,
            y: 0,
            aux: [AUX_UNSEEN; MT_AUX_CODES.len()],
        }
    }
}

/// Authoritative per-slot state, as re-read from the kernel after a
/// SYN_DROPPED (EVIOCGMTSLOTS). `(tracking_id, x, y)`.
pub type SlotSnapshot = [(i32, i32, i32)];

/// Per-slot bookkeeping for scaled (4+ finger) relay.
#[derive(Clone, Copy)]
struct ScaleSlot {
    /// Tracking id the clone was last told for this slot (-1 = inactive).
    clone_id: i32,
    /// Position the clone was last told.
    clone_pos: (i32, i32),
    /// Virtual position accumulator (anchored at touchdown, moves by
    /// real delta * scale).
    virt: (f64, f64),
    /// Real position the last delta was taken from.
    last_real: (i32, i32),
    /// Auxiliary axis values the clone was last told (size/pressure/...,
    /// forwarded UNSCALED -- libinput filters out touches with no size).
    clone_aux: [i32; MT_AUX_CODES.len()],
}

impl Default for ScaleSlot {
    fn default() -> Self {
        ScaleSlot {
            clone_id: -1,
            clone_pos: (0, 0),
            virt: (0.0, 0.0),
            last_real: (0, 0),
            clone_aux: [AUX_UNSEEN; MT_AUX_CODES.len()],
        }
    }
}

pub struct GestureMachine {
    timing: Timing,
    /// Axis extents in pad units, for normalizing flick velocity to
    /// pad-lengths/s per axis.
    x_extent: f64,
    y_extent: f64,
    slot_count: usize,

    slots: [Slot; MAX_SLOTS],
    current_slot: usize,
    /// Which slots the *synthetic clone* currently believes are active.
    relayed_active: [bool; MAX_SLOTS],

    /// True while a 3-finger drag owns the touch: nothing is relayed.
    suppressing: bool,
    drag_ref_slot: Option<usize>,
    drag_last_pos: Option<(i32, i32)>,
    /// While a committed drag hasn't moved yet: when to press the button
    /// anyway (see [`Timing::press_grace`]). Cleared once pressed.
    press_deadline: Option<Instant>,
    /// Diagnostics for the current drag: commit time, total |px| emitted,
    /// and the largest single-frame delta.
    drag_commit_time: Option<Instant>,
    drag_px_total: (i64, i64),
    drag_px_max_frame: i32,

    /// True while the current settled touch is being relayed with
    /// four_finger_scale applied (state-diff relay instead of verbatim).
    scaled_touch: bool,
    /// Smoothed finger velocity (pad-lengths/s) of the scaled touch,
    /// for the flick ramp.
    flick_velocity: f64,
    /// Once a touch reaches full flick speed it stays unscaled for the
    /// REST of the touch: fingers always decelerate before liftoff, and
    /// letting the scale sag back down there would eat the tail of the
    /// gesture's travel -- "once flicked, committed", like macOS.
    flick_latched: bool,
    /// Signed smoothed gesture velocity (units/s per axis), the glide's
    /// launch vector.
    flick_dir: (f64, f64),
    /// Liftoff glide state: the clone's fingers are coasting.
    glide_active: bool,
    glide_v: (f64, f64),
    glide_next: Option<Instant>,
    glide_until: Option<Instant>,
    /// When the previous scaled frame was processed (for velocity dt).
    last_scaled_at: Option<Instant>,
    /// Per-slot scaling state: what the clone was last told (tracking id
    /// and position) plus the virtual (scaled) position accumulator and
    /// the last real position the delta was taken from.
    scale_slots: [ScaleSlot; MAX_SLOTS],

    /// Last EV_KEY values seen from the REAL device (BTN_TOUCH,
    /// BTN_TOOL_*...). The truth about tool state on the pad.
    real_keys: Vec<(u16, i32)>,
    /// Last EV_KEY values the CLONE has been told. Needed to emit
    /// consistent closing state (BTN_TOUCH=0 etc.) when suppression
    /// yanks a partially-relayed touch away -- leaving these stuck at 1
    /// desyncs libinput's finger accounting (and with tap-to-click on,
    /// a slot release without tool release reads as a 2-finger tap:
    /// phantom right-click).
    clone_keys: Vec<(u16, i32)>,

    /// Frames withheld from the compositor while a fresh touch is still
    /// being classified.
    pending: Vec<Ev>,
    touch_start: Option<Instant>,
    touch_max: usize,
    /// Distinct contacts seen across the whole touch (fast, grazing
    /// flicks can have contacts that never overlap in time -- max
    /// simultaneous undercounts them).
    touch_distinct: u32,
    touch_travel: (i64, i64),
    settled: bool,

    /// Virtual left button state (survives across touches for drag-lock).
    held: bool,
    /// When set, the button stays held until this instant unless a new
    /// touch resolves the lock first (see [`Timing::drag_end_delay`]).
    lock_deadline: Option<Instant>,
}

impl GestureMachine {
    pub fn new(timing: Timing, x_extent: f64, y_extent: f64, slot_count: usize) -> Self {
        GestureMachine {
            timing,
            x_extent: x_extent.max(1.0),
            y_extent: y_extent.max(1.0),
            slot_count: slot_count.clamp(1, MAX_SLOTS),
            slots: [Slot::default(); MAX_SLOTS],
            current_slot: 0,
            relayed_active: [false; MAX_SLOTS],
            suppressing: false,
            drag_ref_slot: None,
            drag_last_pos: None,
            press_deadline: None,
            drag_commit_time: None,
            drag_px_total: (0, 0),
            drag_px_max_frame: 0,
            scaled_touch: false,
            flick_velocity: 0.0,
            flick_latched: false,
            flick_dir: (0.0, 0.0),
            glide_active: false,
            glide_v: (0.0, 0.0),
            glide_next: None,
            glide_until: None,
            last_scaled_at: None,
            scale_slots: [ScaleSlot::default(); MAX_SLOTS],
            real_keys: Vec::new(),
            clone_keys: Vec::new(),
            pending: Vec::new(),
            touch_start: None,
            touch_max: 0,
            touch_distinct: 0,
            touch_travel: (0, 0),
            settled: false,
            held: false,
            lock_deadline: None,
        }
    }

    /// Hot-reload hook: swap in new timing/scaling without disturbing
    /// any in-flight touch state.
    pub fn set_timing(&mut self, timing: Timing) {
        self.timing = timing;
    }

    /// Whether the virtual button is currently held (used by the shell
    /// for a defensive release on shutdown).
    pub fn button_held(&self) -> bool {
        self.held
    }

    /// Whether the clone's fingers are coasting after a flick.
    pub fn is_gliding(&self) -> bool {
        self.glide_active
    }

    /// The next instant at which [`on_tick`](Self::on_tick) has work to
    /// do, if any. The I/O loop sleeps exactly until this, so decisions
    /// land on time instead of on the next poll interval.
    pub fn next_deadline(&self) -> Option<Instant> {
        if self.glide_active {
            return self.glide_next;
        }
        if self.suppressing {
            // a committed drag that hasn't moved yet still owes a
            // deferred button press
            return if self.held { None } else { self.press_deadline };
        }
        if let Some(start) = self.touch_start {
            if !self.settled {
                let window = if self.touch_max <= 1 {
                    self.timing.probe_delay
                } else {
                    self.timing.entry_debounce
                };
                return Some(start + window);
            }
            return None; // settled live touch: nothing timed left to decide
        }
        self.lock_deadline
    }

    /// Wall-clock-only work: classification windows closing on a touch
    /// that's holding perfectly still, the deferred drag press, and the
    /// drag-lock timeout.
    pub fn on_tick(&mut self, now: Instant) -> Vec<Output> {
        let mut out = Vec::new();
        if self.glide_active {
            self.step_glide(now, &mut out);
            return out;
        }
        if self.suppressing {
            // Stationary drag: no motion has pressed the button yet, and
            // no 4th finger has shown up to abort -- commit the press.
            if !self.held {
                if let Some(deadline) = self.press_deadline {
                    if now >= deadline {
                        self.press_button(&mut out);
                    }
                }
            }
            return out;
        }

        if self.touch_start.is_none() {
            if let Some(deadline) = self.lock_deadline {
                if now >= deadline {
                    self.lock_deadline = None;
                    self.release_button(&mut out);
                }
            }
            return out;
        }

        if self.settled {
            return out;
        }

        let count = self.active_count();
        let start = self.touch_start.expect("guarded by is_none() above");

        if count == 1 && self.touch_max == 1 && now >= start + self.timing.probe_delay {
            self.settled = true;
            self.settle_live_touch(&mut out);
            return out;
        }
        if now >= start + self.timing.entry_debounce {
            self.resolve_touch_decision(count, now, &mut out);
        }
        out
    }

    /// Feed one complete frame (everything up to and including its
    /// SYN_REPORT). Partial frames interrupted by SYN_DROPPED must not
    /// be fed; discard them and call [`on_resync`](Self::on_resync)
    /// with a fresh kernel snapshot instead.
    pub fn on_frame(&mut self, frame: &[Ev], now: Instant) -> Vec<Output> {
        // 1. fold the frame's slot and key-state updates into our model
        for ev in frame {
            if ev.type_ == EV_KEY {
                Self::note_key(&mut self.real_keys, ev.code, ev.value);
                continue;
            }
            if ev.type_ != EV_ABS {
                continue;
            }
            match ev.code {
                ABS_MT_SLOT => {
                    self.current_slot = (ev.value.max(0) as usize).min(self.slot_count - 1);
                }
                ABS_MT_TRACKING_ID => {
                    if ev.value >= 0 && self.slots[self.current_slot].tracking_id < 0 {
                        self.touch_distinct += 1;
                    }
                    self.slots[self.current_slot].tracking_id = ev.value;
                }
                ABS_MT_POSITION_X => {
                    let s = &mut self.slots[self.current_slot];
                    if s.tracking_id >= 0 {
                        self.touch_travel.0 += i64::from((ev.value - s.x).abs());
                    }
                    s.x = ev.value;
                }
                ABS_MT_POSITION_Y => {
                    let s = &mut self.slots[self.current_slot];
                    if s.tracking_id >= 0 {
                        self.touch_travel.1 += i64::from((ev.value - s.y).abs());
                    }
                    s.y = ev.value;
                }
                code => {
                    if let Some(i) = MT_AUX_CODES.iter().position(|&c| c == code) {
                        self.slots[self.current_slot].aux[i] = ev.value;
                    }
                }
            }
        }
        // 2. decide what to do about it
        let mut out = Vec::new();
        self.decide(frame, now, &mut out);
        out
    }

    /// Reconcile with an authoritative kernel snapshot after the kernel
    /// reported dropped events (SYN_DROPPED). Whatever the dropped
    /// events said is gone; `snapshot` is the truth now.
    pub fn on_resync(&mut self, snapshot: &SlotSnapshot, now: Instant) -> Vec<Output> {
        for slot in 0..MAX_SLOTS {
            self.slots[slot] = match snapshot.get(slot) {
                Some(&(id, x, y)) => Slot {
                    tracking_id: id,
                    x,
                    y,
                    aux: [AUX_UNSEEN; MT_AUX_CODES.len()],
                },
                None => Slot::default(),
            };
        }

        let mut out = Vec::new();

        if self.suppressing {
            // drive_drag reads self.slots directly next frame; the drop
            // may have moved the reference finger arbitrarily far, so
            // re-baseline rather than applying the gap as a cursor jump.
            self.drag_last_pos = None;
        } else if self.touch_start.is_some() && !self.settled && self.active_count() == 0 {
            // Every finger lifted *inside* the dropped window while the
            // touch was still buffered. The buffer holds touchdowns whose
            // matching releases were dropped -- flushing it would leave
            // the synthetic device holding a touch forever. Swallow the
            // (rare) truncated tap instead; consistency wins.
            self.pending.clear();
            self.touch_start = None;
            self.touch_max = 0;
        } else {
            // The synthetic device (or the buffer of a still-undecided
            // touch) may hold stale state. Correct it explicitly: release
            // any slot the clone believes is active but no longer is,
            // then assert the authoritative position of every live slot.
            let mut correction = Vec::new();
            for slot in 0..self.slot_count {
                if self.relayed_active[slot] && self.slots[slot].tracking_id < 0 {
                    correction.push(Ev::abs(ABS_MT_SLOT, slot as i32));
                    correction.push(Ev::abs(ABS_MT_TRACKING_ID, -1));
                }
            }
            correction.extend(self.active_slot_dump());
            if !correction.is_empty() {
                if self.touch_start.is_some() && !self.settled {
                    self.pending.extend(correction);
                    self.pending.push(Ev::syn());
                } else {
                    let mut frame = correction;
                    frame.push(Ev::syn());
                    self.mark_relayed();
                    out.push(Output::EmitSynth(frame));
                }
            }
        }

        // A resync's corrections tell the clone REAL positions; if this
        // touch was being relayed scaled, re-seed the anchors to match.
        if self.scaled_touch {
            self.scaled_touch = false;
            self.maybe_activate_scaling();
        }

        // The dropped events may have included liftoffs (even the whole
        // touch ending). Run the normal decision logic against the new
        // state with an empty frame so we can't be left suppressing (or
        // buffering) a touch that no longer exists.
        self.decide(&[], now, &mut out);
        out
    }

    /// Per-touch diagnostic: what did the REAL pad actually report?
    fn log_touch_autopsy(&mut self, now: Instant, kind: &str) {
        if let Some(t0) = self.touch_start {
            debug!(
                "TOUCH END ({kind}): {}ms, max_simultaneous={}, distinct_contacts={}, travel=|{}|,|{}| units, latched={}",
                now.duration_since(t0).as_millis(),
                self.touch_max,
                self.touch_distinct,
                self.touch_travel.0,
                self.touch_travel.1,
                self.flick_latched,
            );
        }
        self.touch_distinct = 0;
        self.touch_travel = (0, 0);
    }

    // ---- internals ----------------------------------------------------

    fn active_slots(&self) -> Vec<usize> {
        (0..self.slot_count)
            .filter(|&s| self.slots[s].tracking_id >= 0)
            .collect()
    }

    fn active_count(&self) -> usize {
        (0..self.slot_count)
            .filter(|&s| self.slots[s].tracking_id >= 0)
            .count()
    }

    fn decide(&mut self, frame: &[Ev], now: Instant, out: &mut Vec<Output>) {
        let active = self.active_slots();
        let count = active.len();

        // The clone's fingers may still be coasting from a flick: a new
        // real touch needs the clone free immediately, and empty frames
        // from the (empty) pad are none of our business -- the glide
        // owns the clone until it ends.
        if self.glide_active {
            if count == 0 {
                return;
            }
            self.end_glide(out);
        }

        // Once a drag has started, stay suppressed until every finger is
        // off, not just until the count first drops below 3. Fingers
        // never lift in perfect unison; without this hysteresis the
        // trailing 1-2 fingers of a liftoff would be relayed as a fresh
        // touch, which libinput reads as a 2-finger tap (right-click)
        // the moment they lift too.
        if self.suppressing {
            if count == 0 {
                self.log_touch_autopsy(now, "drag");
                if let Some(t0) = self.drag_commit_time.take() {
                    debug!(
                        "DRAG END after {}ms: moved |{}|,|{}| units, max frame delta {} units",
                        now.duration_since(t0).as_millis(),
                        self.drag_px_total.0,
                        self.drag_px_total.1,
                        self.drag_px_max_frame
                    );
                }
                self.suppressing = false;
                self.drag_ref_slot = None;
                self.drag_last_pos = None;
                self.press_deadline = None;
                // Reset touch bookkeeping: without this the next touch
                // would inherit touch_max/settled from this drag and
                // skip the debounce protection entirely.
                self.touch_start = None;
                self.touch_max = 0;
                self.settled = false;
                // A committed drag that never moved and lifted before
                // the press grace still owes its click: press now so the
                // release below (or the drag-lock) completes it.
                self.press_button(out);
                if self.timing.drag_end_delay > Duration::ZERO {
                    // Drag-lock: keep the button held; a new 3-finger
                    // touch inside the window resumes the drag, anything
                    // else releases it (see flush_pending / on_tick).
                    self.lock_deadline = Some(now + self.timing.drag_end_delay);
                } else {
                    self.release_button(out);
                }
                // The synth clone has nothing active on it (suppression
                // never relayed anything), so there's nothing to resync.
                return;
            }
            if count >= 4 {
                // A 4th finger arrived AFTER the entry window closed --
                // this was never a drag, it's a fast/sloppy 4-finger
                // gesture whose last finger staggered in late (the
                // faster the hand comes down, the bigger the stagger).
                // Abort: release the touch to the compositor mid-gesture
                // so the rest of the swipe still registers. Thanks to
                // the deferred press, in the common case no button was
                // ever pressed, so nothing to undo.
                if self.held {
                    warn!(
                        "4th finger after the drag already pressed the button; \
                        releasing (a brief phantom click was unavoidable)"
                    );
                } else {
                    debug!("late 4th finger: aborting committed drag, handing touch to compositor");
                }
                self.suppressing = false;
                self.drag_ref_slot = None;
                self.drag_last_pos = None;
                self.press_deadline = None;
                self.settled = true; // continues as an ordinary live touch
                self.release_button(out);
                self.intro_current_touch(out);
                self.maybe_activate_scaling();
                return;
            }
            self.drive_drag(&active, out);
            // frame intentionally not relayed
            return;
        }

        if count == 0 {
            let had_pending = self.touch_start.is_some() && !self.settled;
            self.log_touch_autopsy(now, if had_pending { "tap" } else { "relayed" });
            self.touch_start = None;
            self.touch_max = 0;
            self.settled = false;
            if had_pending {
                // Touch ended before a decision was reached (e.g. a
                // quick tap): flush everything buffered, including this
                // release frame, so the tap isn't silently swallowed.
                self.pending.extend_from_slice(frame);
                self.flush_pending(out);
                return;
            }
            // Already-settled touch ending (most touches): this frame
            // carries the release events the compositor needs to see.
            if self.scaled_touch {
                if self.flick_latched {
                    // Inertia: don't release the clone's fingers -- let
                    // them coast (see on_tick) so the gesture receives
                    // the travel the flick physically implied.
                    self.start_glide(now);
                } else {
                    self.sync_scaled(now, out); // diff emits releases + key zeros
                    self.reset_scaled_state();
                }
            } else {
                self.relay_frame(frame, out);
            }
            return;
        }

        if self.touch_start.is_none() {
            // the first frame of a brand new touch
            self.touch_start = Some(now);
            self.touch_max = count;
            self.touch_distinct = count as u32;
            self.touch_travel = (0, 0);
            self.settled = false;
            self.pending.clear();
        } else {
            self.touch_max = self.touch_max.max(count);
        }

        if self.settled {
            // Already decided this touch is an ordinary gesture -- relay
            // live. One exception: a touch that *grew* to exactly 3
            // without ever lifting (1->2->3 well after the window
            // closed) is a deliberate late drag and must not leak
            // through as a real 3-finger touch. The touch_max == 3
            // guard is what keeps the *other* way of hitting count == 3
            // -- a 4-finger gesture shedding a finger (4->3), which
            // happens at the tail of every 4-finger swipe because
            // fingers never lift in unison -- from being hijacked into
            // a phantom drag + click.
            if count == 3 && self.touch_max == 3 {
                self.commit_drag(&active, now, out);
                return;
            }
            self.maybe_activate_scaling();
            if self.scaled_touch {
                self.sync_scaled(now, out);
            } else {
                self.relay_frame(frame, out);
            }
            return;
        }

        self.pending.extend_from_slice(frame);

        if self.touch_max >= 4 {
            // Unambiguously bigger than a 3-finger drag could ever be --
            // no need to wait out the rest of the window.
            self.settled = true;
            self.settle_live_touch(out);
            return;
        }

        let start = self.touch_start.expect("set above when the touch began");

        if count == 1 && self.touch_max == 1 && now >= start + self.timing.probe_delay {
            // Still just one finger after a short probe: ordinary
            // pointer movement, by far the most common case. Go live now
            // rather than waiting out the full entry_debounce, or every
            // touch-lift-reposition cycle of normal cursor use would add
            // a felt hitch.
            self.settled = true;
            self.settle_live_touch(out);
            return;
        }

        if now >= start + self.timing.entry_debounce {
            self.resolve_touch_decision(count, now, out);
        }
    }

    /// The entry_debounce window has closed: commit to a drag if the
    /// touch held stably at exactly 3 fingers the whole time, otherwise
    /// release it to the compositor as an ordinary gesture.
    fn resolve_touch_decision(&mut self, count: usize, now: Instant, out: &mut Vec<Output>) {
        if count == 3 && self.touch_max == 3 {
            self.pending.clear();
            let active = self.active_slots();
            self.commit_drag(&active, now, out);
            return;
        }
        self.settled = true;
        self.settle_live_touch(out);
    }

    /// Commit the current touch as a 3-finger drag. The button press is
    /// DEFERRED: it fires at the first actual drag motion, or when
    /// press_grace expires -- so a late 4th finger (fast 4-finger swipe)
    /// can still abort without a phantom click having been sent.
    fn commit_drag(&mut self, active: &[usize], now: Instant, out: &mut Vec<Output>) {
        let elapsed = self
            .touch_start
            .map(|t| now.duration_since(t).as_millis())
            .unwrap_or(0);
        let contacts: Vec<String> = active
            .iter()
            .map(|&s| {
                format!(
                    "slot{}(id={} x={} y={} major={})",
                    s,
                    self.slots[s].tracking_id,
                    self.slots[s].x,
                    self.slots[s].y,
                    self.slots[s].touch_major()
                )
            })
            .collect();
        debug!(
            "DRAG COMMIT after {}ms (touch_max={}): {}",
            elapsed,
            self.touch_max,
            contacts.join(" ")
        );
        self.drag_commit_time = Some(now);
        self.drag_px_total = (0, 0);
        self.drag_px_max_frame = 0;
        self.settled = true;
        self.enter_suppress(out);
        if !self.held {
            self.press_deadline = Some(now + self.timing.press_grace);
        }
        self.drive_drag(active, out);
    }

    /// Releases a buffered touch to the compositor: it either never
    /// became a drag, or grew past 3 into a bigger gesture that isn't
    /// ours to intercept. If a drag-lock is pending, the button is
    /// released *first*, so the flushed motion can never drag anything.
    fn flush_pending(&mut self, out: &mut Vec<Output>) {
        if self.lock_deadline.take().is_some() {
            self.release_button(out);
        }
        if !self.pending.is_empty() {
            let flushed = std::mem::take(&mut self.pending);
            self.note_clone_keys(&flushed);
            out.push(Output::EmitSynth(flushed));
        }
        self.mark_relayed();
    }

    /// Introduce the current, live touch to the clone as a fresh,
    /// complete, consistent touchdown: every live slot at its CURRENT
    /// position, plus the real pad's current tool state
    /// (BTN_TOUCH/BTN_TOOL_*).
    fn intro_current_touch(&mut self, out: &mut Vec<Output>) {
        let mut intro = self.active_slot_dump();
        for i in 0..self.real_keys.len() {
            let (code, value) = self.real_keys[i];
            if Self::key_value(&self.clone_keys, code) != value {
                intro.push(Ev::new(EV_KEY, code, value));
                Self::note_key(&mut self.clone_keys, code, value);
            }
        }
        intro.push(Ev::syn());
        self.mark_relayed();
        out.push(Output::EmitSynth(intro));
    }

    /// A still-alive touch has settled as "not ours": hand it to the
    /// compositor as a fresh touchdown at its CURRENT state, discarding
    /// the buffered history. Replaying the buffer would deliver all its
    /// motion in one burst -- uinput regenerates timestamps, so libinput
    /// sees the buffered movement at near-infinite velocity, the accel
    /// curve maxes out, and gestures (especially fast 4-finger swipes)
    /// leap at onset. A quick tap that already ENDED still uses
    /// flush_pending: its verbatim replay is what keeps tap gestures
    /// working, and taps carry no meaningful motion to burst.
    fn settle_live_touch(&mut self, out: &mut Vec<Output>) {
        if self.lock_deadline.take().is_some() {
            self.release_button(out);
        }
        self.pending.clear();
        self.intro_current_touch(out);
        self.maybe_activate_scaling();
    }

    /// Activate scaled relay for the current settled touch if it
    /// qualifies (4+ fingers, scale configured below 1.0). Seeds the
    /// per-slot anchors from the CURRENT real state -- which is what the
    /// clone was just told, whether by intro or by verbatim relay.
    fn maybe_activate_scaling(&mut self) {
        if self.scaled_touch || self.timing.four_finger_scale >= 1.0 || self.touch_max < 4 {
            return;
        }
        for slot in 0..self.slot_count {
            let real = self.slots[slot];
            let ss = &mut self.scale_slots[slot];
            // Seed only from what the clone has actually been told
            // (relayed_active): in the growth case (2 -> 4 fingers)
            // activation happens on the frame that ADDS fingers, before
            // any relay -- those must stay unknown here so the first
            // sync introduces them properly.
            if real.tracking_id >= 0 && self.relayed_active[slot] {
                ss.clone_id = real.tracking_id;
                ss.clone_pos = (real.x, real.y);
                ss.virt = (real.x as f64, real.y as f64);
                ss.last_real = (real.x, real.y);
                ss.clone_aux = real.aux;
            } else {
                *ss = ScaleSlot::default();
            }
        }
        self.scaled_touch = true;
        self.flick_velocity = 0.0;
        self.flick_latched = false;
        self.last_scaled_at = None;
        debug!(
            "4+ finger touch: relaying with motion scale {} (flick ramp to 1.0 above {} pad-lengths/s)",
            self.timing.four_finger_scale, FLICK_HI_PADS_S
        );
    }

    /// Scaled relay: instead of forwarding the raw frame, emit a state
    /// DIFF between what the clone knows and the real pad, with each
    /// finger's motion scaled around its touchdown anchor. Because the
    /// diff is computed against our authoritative slot model, slot
    /// context on the clone can never desync.
    fn sync_scaled(&mut self, now: Instant, out: &mut Vec<Output>) {
        // Velocity-adaptive scale: mean per-frame finger travel,
        // normalized to mm/s and smoothed, ramps the effective scale
        // from four_finger_scale (slow, precise) up to 1.0 (a flick --
        // full travel, so compositor gestures complete with momentum).
        let dt_s = self
            .last_scaled_at
            .map(|t| now.duration_since(t).as_secs_f64().clamp(0.004, 0.1));
        self.last_scaled_at = Some(now);
        if let Some(dt_s) = dt_s {
            // (the first frame after activation only seeds the clock --
            // there is no baseline to take a velocity sample against)
            let (mut travel_x, mut travel_y, mut signed_x, mut signed_y, mut moving) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0u32);
            for slot in 0..self.slot_count {
                let real = self.slots[slot];
                let ss = &self.scale_slots[slot];
                if real.tracking_id >= 0 && ss.clone_id == real.tracking_id {
                    travel_x += f64::from((real.x - ss.last_real.0).abs());
                    travel_y += f64::from((real.y - ss.last_real.1).abs());
                    signed_x += f64::from(real.x - ss.last_real.0);
                    signed_y += f64::from(real.y - ss.last_real.1);
                    moving += 1;
                }
            }
            if moving > 0 {
                let vx = travel_x / f64::from(moving) / self.x_extent / dt_s;
                let vy = travel_y / f64::from(moving) / self.y_extent / dt_s;
                let v = vx.max(vy);
                // weight new samples heavily: a flick's fastest frames
                // are its FIRST frames, and a slow estimator burns them
                // at the calm scale
                self.flick_velocity = 0.3 * self.flick_velocity + 0.7 * v;
                self.flick_dir.0 =
                    0.3 * self.flick_dir.0 + 0.7 * (signed_x / f64::from(moving) / dt_s);
                self.flick_dir.1 =
                    0.3 * self.flick_dir.1 + 0.7 * (signed_y / f64::from(moving) / dt_s);
            }
        }
        let t = ((self.flick_velocity - FLICK_LO_PADS_S) / (FLICK_HI_PADS_S - FLICK_LO_PADS_S))
            .clamp(0.0, 1.0);
        if t >= 1.0 && !self.flick_latched {
            self.flick_latched = true;
            // Retroactive momentum: restore the travel withheld while
            // the ramp was still recognizing the flick. Snapping each
            // virtual finger to its last real position (the pending
            // frame's delta is applied unscaled below) hands the
            // compositor the flick's FULL physical travel -- without
            // this, the scaled-down early frames leave short-axis
            // gestures short of their completion distance.
            for slot in 0..self.slot_count {
                let real = self.slots[slot];
                let ss = &mut self.scale_slots[slot];
                if real.tracking_id >= 0 && ss.clone_id == real.tracking_id {
                    ss.virt = (f64::from(ss.last_real.0), f64::from(ss.last_real.1));
                }
            }
        }
        let scale = if self.flick_latched {
            1.0
        } else {
            let t = t * t * (3.0 - 2.0 * t); // smoothstep
            let base = self.timing.four_finger_scale;
            base + (1.0 - base) * t
        };

        let mut frame = Vec::new();
        for slot in 0..self.slot_count {
            let real = self.slots[slot];
            let ss = &mut self.scale_slots[slot];
            if real.tracking_id >= 0 {
                if ss.clone_id != real.tracking_id {
                    // finger landed (or id changed): anchor here
                    ss.clone_id = real.tracking_id;
                    ss.clone_pos = (real.x, real.y);
                    ss.virt = (real.x as f64, real.y as f64);
                    ss.last_real = (real.x, real.y);
                    frame.push(Ev::abs(ABS_MT_SLOT, slot as i32));
                    frame.push(Ev::abs(ABS_MT_TRACKING_ID, real.tracking_id));
                    frame.push(Ev::abs(ABS_MT_POSITION_X, real.x));
                    frame.push(Ev::abs(ABS_MT_POSITION_Y, real.y));
                    for (i, &code) in MT_AUX_CODES.iter().enumerate() {
                        if real.aux[i] != AUX_UNSEEN {
                            frame.push(Ev::abs(code, real.aux[i]));
                            ss.clone_aux[i] = real.aux[i];
                        }
                    }
                } else {
                    ss.virt.0 += f64::from(real.x - ss.last_real.0) * scale;
                    ss.virt.1 += f64::from(real.y - ss.last_real.1) * scale;
                    ss.last_real = (real.x, real.y);
                    let v = (ss.virt.0.round() as i32, ss.virt.1.round() as i32);
                    let aux_changed = (0..MT_AUX_CODES.len())
                        .any(|i| real.aux[i] != AUX_UNSEEN && real.aux[i] != ss.clone_aux[i]);
                    if v != ss.clone_pos || aux_changed {
                        frame.push(Ev::abs(ABS_MT_SLOT, slot as i32));
                        if v.0 != ss.clone_pos.0 {
                            frame.push(Ev::abs(ABS_MT_POSITION_X, v.0));
                        }
                        if v.1 != ss.clone_pos.1 {
                            frame.push(Ev::abs(ABS_MT_POSITION_Y, v.1));
                        }
                        for (i, &code) in MT_AUX_CODES.iter().enumerate() {
                            if real.aux[i] != AUX_UNSEEN && real.aux[i] != ss.clone_aux[i] {
                                frame.push(Ev::abs(code, real.aux[i]));
                                ss.clone_aux[i] = real.aux[i];
                            }
                        }
                        ss.clone_pos = v;
                    }
                }
            } else if ss.clone_id >= 0 {
                *ss = ScaleSlot::default();
                frame.push(Ev::abs(ABS_MT_SLOT, slot as i32));
                frame.push(Ev::abs(ABS_MT_TRACKING_ID, -1));
            }
        }
        for i in 0..self.real_keys.len() {
            let (code, value) = self.real_keys[i];
            if Self::key_value(&self.clone_keys, code) != value {
                frame.push(Ev::new(EV_KEY, code, value));
                Self::note_key(&mut self.clone_keys, code, value);
            }
        }
        self.mark_relayed();
        if !frame.is_empty() {
            frame.push(Ev::syn());
            out.push(Output::EmitSynth(frame));
        }
    }

    fn reset_scaled_state(&mut self) {
        self.scaled_touch = false;
        self.flick_velocity = 0.0;
        self.flick_latched = false;
        self.flick_dir = (0.0, 0.0);
        self.last_scaled_at = None;
        self.scale_slots = [ScaleSlot::default(); MAX_SLOTS];
    }

    fn start_glide(&mut self, now: Instant) {
        debug!(
            "GLIDE start: v=({:.0},{:.0}) units/s",
            self.flick_dir.0, self.flick_dir.1
        );
        self.glide_active = true;
        self.glide_v = self.flick_dir;
        self.glide_next = Some(now + GLIDE_STEP);
        self.glide_until = Some(now + GLIDE_MAX);
        // scaled_touch stays true only in the sense that scale_slots
        // still describe the clone; classification state resets so the
        // next real touch is judged from scratch
        self.scaled_touch = false;
        self.flick_velocity = 0.0;
        self.flick_latched = false;
        self.last_scaled_at = None;
    }

    /// One coasting step: advance every still-active clone finger along
    /// the (decaying) launch vector.
    fn step_glide(&mut self, now: Instant, out: &mut Vec<Output>) {
        let Some(mut next) = self.glide_next else {
            return;
        };
        let until = self.glide_until.unwrap_or(now);
        let mut frame = Vec::new();
        while now >= next {
            let dt = GLIDE_STEP.as_secs_f64();
            for slot in 0..self.slot_count {
                let ss = &mut self.scale_slots[slot];
                if ss.clone_id >= 0 {
                    ss.virt.0 += self.glide_v.0 * dt;
                    ss.virt.1 += self.glide_v.1 * dt;
                    let v = (ss.virt.0.round() as i32, ss.virt.1.round() as i32);
                    if v != ss.clone_pos {
                        frame.push(Ev::abs(ABS_MT_SLOT, slot as i32));
                        if v.0 != ss.clone_pos.0 {
                            frame.push(Ev::abs(ABS_MT_POSITION_X, v.0));
                        }
                        if v.1 != ss.clone_pos.1 {
                            frame.push(Ev::abs(ABS_MT_POSITION_Y, v.1));
                        }
                        ss.clone_pos = v;
                    }
                }
            }
            self.glide_v.0 *= GLIDE_DECAY;
            self.glide_v.1 *= GLIDE_DECAY;
            next += GLIDE_STEP;

            let speed_pads =
                (self.glide_v.0.abs() / self.x_extent).max(self.glide_v.1.abs() / self.y_extent);
            if speed_pads < GLIDE_MIN_PADS_S || next > until {
                if !frame.is_empty() {
                    frame.push(Ev::syn());
                    out.push(Output::EmitSynth(std::mem::take(&mut frame)));
                }
                self.end_glide(out);
                return;
            }
        }
        self.glide_next = Some(next);
        if !frame.is_empty() {
            frame.push(Ev::syn());
            out.push(Output::EmitSynth(frame));
        }
    }

    /// Release the coasting fingers and their tool state.
    fn end_glide(&mut self, out: &mut Vec<Output>) {
        let mut frame = Vec::new();
        for slot in 0..self.slot_count {
            if self.scale_slots[slot].clone_id >= 0 {
                frame.push(Ev::abs(ABS_MT_SLOT, slot as i32));
                frame.push(Ev::abs(ABS_MT_TRACKING_ID, -1));
            }
        }
        for i in 0..self.clone_keys.len() {
            if self.clone_keys[i].1 != 0 {
                frame.push(Ev::new(EV_KEY, self.clone_keys[i].0, 0));
                self.clone_keys[i].1 = 0;
            }
        }
        if !frame.is_empty() {
            frame.push(Ev::syn());
            out.push(Output::EmitSynth(frame));
        }
        self.glide_active = false;
        self.glide_next = None;
        self.glide_until = None;
        self.glide_v = (0.0, 0.0);
        self.reset_scaled_state();
        // the clone is now empty -- NOT mark_relayed(): a new real touch
        // may already be down (and still buffered, unseen by the clone)
        self.relayed_active = [false; MAX_SLOTS];
        debug!("GLIDE end: fingers released");
    }

    fn relay_frame(&mut self, frame: &[Ev], out: &mut Vec<Output>) {
        self.mark_relayed();
        if !frame.is_empty() {
            self.note_clone_keys(frame);
            out.push(Output::EmitSynth(frame.to_vec()));
        }
    }

    fn mark_relayed(&mut self) {
        for slot in 0..MAX_SLOTS {
            self.relayed_active[slot] = self.slots[slot].tracking_id >= 0;
        }
    }

    /// Record what EV_KEY state a batch of events tells the clone.
    fn note_clone_keys(&mut self, events: &[Ev]) {
        for ev in events {
            if ev.type_ == EV_KEY {
                Self::note_key(&mut self.clone_keys, ev.code, ev.value);
            }
        }
    }

    fn note_key(map: &mut Vec<(u16, i32)>, code: u16, value: i32) {
        for entry in map.iter_mut() {
            if entry.0 == code {
                entry.1 = value;
                return;
            }
        }
        map.push((code, value));
    }

    fn key_value(map: &[(u16, i32)], code: u16) -> i32 {
        map.iter().find(|e| e.0 == code).map(|e| e.1).unwrap_or(0)
    }

    /// Begin withholding everything from the compositor. Any slots the
    /// synthetic clone still believes are active are explicitly released
    /// first -- and any tool state (BTN_TOUCH, BTN_TOOL_*) it was told is
    /// pressed is explicitly released too, so it is never left holding a
    /// half-open touch or stuck finger-count bits (which desync
    /// libinput's tap/gesture accounting).
    fn enter_suppress(&mut self, out: &mut Vec<Output>) {
        self.suppressing = true;
        self.lock_deadline = None; // a live drag owns the button now

        let mut release = Vec::new();
        for slot in 0..MAX_SLOTS {
            if self.relayed_active[slot] {
                release.push(Ev::abs(ABS_MT_SLOT, slot as i32));
                release.push(Ev::abs(ABS_MT_TRACKING_ID, -1));
                self.relayed_active[slot] = false;
            }
        }
        for i in 0..self.clone_keys.len() {
            let (code, value) = self.clone_keys[i];
            if value != 0 {
                release.push(Ev::new(EV_KEY, code, 0));
                self.clone_keys[i].1 = 0;
            }
        }
        if !release.is_empty() {
            release.push(Ev::syn());
            out.push(Output::EmitSynth(release));
        }
    }

    fn press_button(&mut self, out: &mut Vec<Output>) {
        if !self.held {
            self.held = true;
            self.press_deadline = None;
            out.push(Output::MouseDown);
        }
        // if still held from a drag-lock, the drag just resumes --
        // no re-press, no glitch
    }

    fn release_button(&mut self, out: &mut Vec<Output>) {
        if self.held {
            self.held = false;
            out.push(Output::MouseUp);
        }
    }

    fn drive_drag(&mut self, active: &[usize], out: &mut Vec<Output>) {
        let reference = match self.drag_ref_slot {
            Some(s) if active.contains(&s) => s,
            _ => {
                // first frame of the gesture, or the previous reference
                // finger lifted and another took its place: re-baseline
                // without applying a delta this frame.
                let s = active[0];
                self.drag_ref_slot = Some(s);
                self.drag_last_pos = Some((self.slots[s].x, self.slots[s].y));
                return;
            }
        };

        let (x, y) = (self.slots[reference].x, self.slots[reference].y);
        if let Some((lx, ly)) = self.drag_last_pos {
            let (dx, dy) = (x - lx, y - ly);
            if dx != 0 || dy != 0 {
                // real drag motion: the deferred press (if still pending)
                // must land before the movement it accompanies
                self.press_button(out);
                self.drag_px_total.0 += i64::from(dx.unsigned_abs());
                self.drag_px_total.1 += i64::from(dy.unsigned_abs());
                self.drag_px_max_frame = self.drag_px_max_frame.max(dx.abs().max(dy.abs()));
                out.push(Output::MouseMove { dx, dy });
            }
        } else {
            self.drag_last_pos = Some((x, y));
            return;
        }
        self.drag_last_pos = Some((x, y));
    }

    /// SLOT/TRACKING_ID/X/Y events asserting the current state of every
    /// active slot (used to correct downstream state after a resync).
    fn active_slot_dump(&self) -> Vec<Ev> {
        let mut dump = Vec::new();
        for slot in 0..self.slot_count {
            let s = self.slots[slot];
            if s.tracking_id >= 0 {
                dump.push(Ev::abs(ABS_MT_SLOT, slot as i32));
                dump.push(Ev::abs(ABS_MT_TRACKING_ID, s.tracking_id));
                dump.push(Ev::abs(ABS_MT_POSITION_X, s.x));
                dump.push(Ev::abs(ABS_MT_POSITION_Y, s.y));
                for (i, &code) in MT_AUX_CODES.iter().enumerate() {
                    if s.aux[i] != AUX_UNSEEN {
                        dump.push(Ev::abs(code, s.aux[i]));
                    }
                }
            }
        }
        dump
    }
}

#[cfg(test)]
mod fuzz;
#[cfg(test)]
mod tests;
