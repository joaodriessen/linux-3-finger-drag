// Proxies the real touchpad's raw multitouch stream to a synthetic clone
// device, 1:1, EXCEPT while exactly 3 fingers are touching: that window is
// withheld from the clone entirely (never begun, never left half-open) and
// instead drives the existing 3-finger-drag emulation on the virtual mouse.
//
// This exists because a naive mid-gesture EVIOCGRAB on the real device (this
// project's first approach) leaves the compositor's own touch-state tracking
// permanently corrupted: the compositor never gets to see the closing
// lift-off frames for the fingers grabbed away mid-gesture, so it's stuck
// believing they're still down. By instead owning the real device for the
// program's entire lifetime and re-emitting a clean copy of it, we control
// every frame the compositor ever sees: either an accurate mirror of the
// real pad, or an explicit "nothing is touching" state we emit ourselves --
// never a silent gap.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::time::Instant;

use nix::libc::O_NONBLOCK;
use tracing::{debug, info, trace, warn};

use input_linux::{sys, AbsoluteAxis, AbsoluteInfoSetup, EvdevHandle, EventKind, UInputHandle};

use super::event_handler::{GestureTranslator, GtError};

const EV_SYN: u16 = 0x00;
const EV_ABS: u16 = 0x03;
const SYN_REPORT: u16 = 0x00;
const SYN_DROPPED: u16 = 0x03;
const ABS_MT_SLOT: u16 = 0x2f;
const ABS_MT_TRACKING_ID: u16 = 0x39;
const ABS_MT_POSITION_X: u16 = 0x35;
const ABS_MT_POSITION_Y: u16 = 0x36;

const MAX_SLOTS: usize = 16;
const READ_BATCH: usize = 64;

// px-per-mm scale for turning the real finger delta into cursor movement;
// combines with the `acceleration` config knob on top. 12.0 (4.0 x the
// initial guess) is the value the user confirmed feels right live.
const PX_PER_MM: f64 = 12.0;

#[derive(Clone, Copy)]
struct Slot {
    tracking_id: i32,
    x: i32,
    y: i32,
}

impl Default for Slot {
    fn default() -> Self {
        Slot { tracking_id: -1, x: 0, y: 0 }
    }
}

fn zero_event() -> sys::input_event {
    unsafe { std::mem::zeroed() }
}

fn abs_event(code: u16, value: i32) -> sys::input_event {
    let mut ev = zero_event();
    ev.type_ = EV_ABS;
    ev.code = code;
    ev.value = value;
    ev
}

fn syn_report() -> sys::input_event {
    let mut ev = zero_event();
    ev.type_ = EV_SYN;
    ev.code = SYN_REPORT;
    ev.value = 0;
    ev
}

pub struct MtProxy {
    real: EvdevHandle<File>,
    synth: UInputHandle<File>,
    x_res: f64,
    y_res: f64,
    slots: [Slot; MAX_SLOTS],
    current_slot: usize,
    relayed_active: [bool; MAX_SLOTS],
    suppressing: bool,
    drag_ref_slot: Option<usize>,
    drag_last_pos: Option<(i32, i32)>,
    frame: Vec<sys::input_event>,
    read_buf: [sys::input_event; READ_BATCH],
    // Bookkeeping for one continuous touch (from the first finger down to
    // all fingers up), tracked regardless of how many fingers are on the
    // pad. Frames are buffered -- withheld from the compositor entirely --
    // until this touch is either "settled" (decided to be an ordinary
    // gesture, safe to relay live from here on) or turns into a drag. See
    // handle_frame for the decision logic.
    pending_frames: Vec<sys::input_event>,
    touch_start: Option<Instant>,
    touch_max: usize,
    settled: bool,
}

impl MtProxy {
    /// Opens the real touchpad at `path`, grabs it exclusively for the rest
    /// of the program's life, and creates a synthetic clone with the same
    /// multitouch capabilities for the compositor to read instead.
    pub fn new(path: &str) -> io::Result<Self> {
        let real_file = OpenOptions::new()
            .read(true)
            .custom_flags(O_NONBLOCK)
            .open(path)?;
        let real = EvdevHandle::new(real_file);

        // held for the entire program lifetime -- see module doc for why
        // this must never be released mid-gesture.
        real.grab(true)?;
        info!("Exclusively grabbed the real trackpad at {}.", path);

        let synth = Self::clone_device(&real)?;

        let x_res = real.absolute_info(AbsoluteAxis::MultitouchPositionX)
            .map(|i| i.resolution.max(1) as f64)
            .unwrap_or(1.0);
        let y_res = real.absolute_info(AbsoluteAxis::MultitouchPositionY)
            .map(|i| i.resolution.max(1) as f64)
            .unwrap_or(1.0);

        Ok(MtProxy {
            real,
            synth,
            x_res,
            y_res,
            slots: [Slot::default(); MAX_SLOTS],
            current_slot: 0,
            relayed_active: [false; MAX_SLOTS],
            suppressing: false,
            drag_ref_slot: None,
            drag_last_pos: None,
            frame: Vec::with_capacity(READ_BATCH),
            read_buf: [zero_event(); READ_BATCH],
            pending_frames: Vec::new(),
            touch_start: None,
            touch_max: 0,
            settled: false,
        })
    }

    /// Builds a synthetic uinput device with the same EV_KEY/EV_ABS/
    /// INPUT_PROP capabilities as the real device, so the compositor's own
    /// libinput sees something functionally identical to the real hardware.
    fn clone_device(real: &EvdevHandle<File>) -> io::Result<UInputHandle<File>> {
        let uinput_file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(O_NONBLOCK)
            .open("/dev/uinput")?;
        let synth = UInputHandle::new(uinput_file);

        synth.set_evbit(EventKind::Key)?;
        synth.set_evbit(EventKind::Absolute)?;

        for key in real.key_bits()?.iter() {
            synth.set_keybit(key)?;
        }

        let mut abs_setups = Vec::new();
        for axis in real.absolute_bits()?.iter() {
            synth.set_absbit(axis)?;
            let info = real.absolute_info(axis)?;
            abs_setups.push(AbsoluteInfoSetup { axis, info });
        }

        for prop in real.device_properties()?.iter() {
            synth.set_propbit(prop)?;
        }

        // Impersonate the real device's identity (vendor/product/name), not
        // just its capabilities. KDE keys its per-device libinput settings
        // (natural scroll, pointer accel profile, tap-to-click, click
        // method...) in kcminputrc by exactly this triple -- a synthetic
        // device with a made-up identity is "new" to KDE and silently falls
        // back to defaults, which is what caused scrolling to come back
        // reversed after this proxy replaced the real device as KWin's
        // input source. Matching identity means the user's existing saved
        // preferences apply automatically, with nothing to keep in sync.
        let real_id = real.device_id()?;
        let mut real_name = real.device_name()?;
        while real_name.last() == Some(&0) {
            real_name.pop();
        }
        synth.create(&real_id, &real_name, 0, &abs_setups)?;
        debug!("Synthetic touchpad clone created, impersonating \"{}\".", String::from_utf8_lossy(&real_name));

        std::thread::sleep(std::time::Duration::from_millis(500));

        Ok(synth)
    }

    /// One non-blocking pass: drains whatever raw events are currently
    /// available, processing them frame-by-frame (a frame being everything
    /// since the last SYN_REPORT). Safe to call frequently in a poll loop.
    pub async fn poll(&mut self, translator: &mut GestureTranslator) -> Result<(), GtError> {
        loop {
            // Independent of whether new events arrived this tick: a touch
            // that's holding perfectly still (no new events at all) still
            // needs its probe/confirm decision made on wall-clock time.
            self.resolve_touch_timeout(translator).await?;

            let n = match self.real.read(&mut self.read_buf) {
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(e) => return Err(GtError::from(e)),
            };

            if n == 0 {
                return Ok(());
            }

            for i in 0..n {
                let ev = self.read_buf[i];

                if ev.type_ == EV_SYN && ev.code == SYN_DROPPED {
                    warn!("Kernel reported dropped events; resyncing slot state.");
                    self.frame.clear();
                    self.resync()?;
                    continue;
                }

                if ev.type_ == EV_SYN && ev.code == SYN_REPORT {
                    self.frame.push(ev);
                    self.handle_frame(translator).await?;
                    self.frame.clear();
                    continue;
                }

                if ev.type_ == EV_ABS && ev.code == ABS_MT_SLOT {
                    self.current_slot = (ev.value as usize).min(MAX_SLOTS - 1);
                } else if ev.type_ == EV_ABS && ev.code == ABS_MT_TRACKING_ID {
                    self.slots[self.current_slot].tracking_id = ev.value;
                } else if ev.type_ == EV_ABS && ev.code == ABS_MT_POSITION_X {
                    self.slots[self.current_slot].x = ev.value;
                } else if ev.type_ == EV_ABS && ev.code == ABS_MT_POSITION_Y {
                    self.slots[self.current_slot].y = ev.value;
                }

                self.frame.push(ev);
            }
        }
    }

    /// Re-derives slot state directly from the kernel (EVIOCGMTSLOTS)
    /// instead of trusting the incremental event history, used after a
    /// SYN_DROPPED.
    fn resync(&mut self) -> Result<(), GtError> {
        let mut ids = vec![0i32; MAX_SLOTS];
        self.real.multi_touch_slots(AbsoluteAxis::MultitouchTrackingId, &mut ids)?;

        let mut xs = vec![0i32; MAX_SLOTS];
        self.real.multi_touch_slots(AbsoluteAxis::MultitouchPositionX, &mut xs)?;

        let mut ys = vec![0i32; MAX_SLOTS];
        self.real.multi_touch_slots(AbsoluteAxis::MultitouchPositionY, &mut ys)?;

        for slot in 0..MAX_SLOTS {
            self.slots[slot] = Slot { tracking_id: ids[slot], x: xs[slot], y: ys[slot] };
        }

        // Whatever events the kernel dropped may have left the synthetic
        // device (or a frame we're still buffering, waiting to decide what
        // to do with it) holding stale positions for slots that are still
        // active. Left uncorrected, a later real event touching that slot
        // again looks like a sudden, discontinuous jump once it finally
        // arrives -- exactly the kind of anomaly a pinch/zoom gesture
        // recognizer can misfire on. So assert the now-authoritative
        // position explicitly rather than waiting for it to resolve
        // itself.
        let correction = self.active_slot_dump();
        if correction.is_empty() {
            return Ok(());
        }

        if self.suppressing {
            // drive_drag reads self.slots directly next frame, and nothing
            // was relayed during suppression, so there's nothing
            // downstream to correct.
        } else if self.touch_start.is_some() && !self.settled {
            // Still deciding this touch's fate -- fold the correction into
            // the buffer so it's included whenever this touch is flushed.
            self.pending_frames.extend(correction);
            self.pending_frames.push(syn_report());
        } else {
            // Live passthrough: the gap is visible right now, so correct
            // it immediately instead of waiting for the next real frame.
            let mut frame = correction;
            frame.push(syn_report());
            self.synth.write(&frame)?;
            for slot in 0..MAX_SLOTS {
                self.relayed_active[slot] = self.slots[slot].tracking_id >= 0;
            }
        }

        // don't decide suppress/passthrough here; the next real frame will
        // trigger handle_frame() and pick correctly based on active_count
        Ok(())
    }

    /// Builds SLOT/TRACKING_ID/POSITION_X/POSITION_Y events asserting the
    /// current, authoritative state of every active slot.
    fn active_slot_dump(&self) -> Vec<sys::input_event> {
        let mut dump = Vec::new();
        for slot in 0..MAX_SLOTS {
            let s = self.slots[slot];
            if s.tracking_id >= 0 {
                dump.push(abs_event(ABS_MT_SLOT, slot as i32));
                dump.push(abs_event(ABS_MT_TRACKING_ID, s.tracking_id));
                dump.push(abs_event(ABS_MT_POSITION_X, s.x));
                dump.push(abs_event(ABS_MT_POSITION_Y, s.y));
            }
        }
        dump
    }

    fn active_slots(&self) -> Vec<usize> {
        (0..MAX_SLOTS).filter(|&s| self.slots[s].tracking_id >= 0).collect()
    }

    async fn handle_frame(&mut self, translator: &mut GestureTranslator) -> Result<(), GtError> {
        let active = self.active_slots();
        let count = active.len();

        // Once a drag has started, stay suppressed until every finger is
        // off, not just until the count first drops below 3. Fingers never
        // lift in perfect unison; without this hysteresis the trailing 1-2
        // fingers of a liftoff get relayed as a fresh, real touch the
        // instant the first finger leaves -- which libinput reads as a
        // brief 2-finger tap (right-click) the moment the rest lift too.
        if self.suppressing {
            if count == 0 {
                self.suppressing = false;
                self.drag_ref_slot = None;
                self.drag_last_pos = None;
                // Reset touch bookkeeping too: without this, the next
                // touch inherits touch_max == 3 / settled == true from
                // this drag, so a later stray 3-finger moment would skip
                // the debounce protection entirely.
                self.touch_start = None;
                self.touch_max = 0;
                self.settled = false;
                translator.handle_mouse_up().await?;
                // synth already has nothing active on it (suppression never
                // relayed anything), so there's nothing to resync here
                return Ok(());
            }
            self.drive_drag(&active, translator).await?;
            // frame intentionally not relayed
            return Ok(());
        }

        if count == 0 {
            let had_pending = self.touch_start.is_some() && !self.settled;
            self.touch_start = None;
            self.touch_max = 0;
            self.settled = false;
            if had_pending {
                // Touch ended before a decision was ever reached (e.g. a
                // quick tap) -- flush whatever was buffered, including this
                // release frame, so it isn't silently swallowed.
                self.pending_frames.extend_from_slice(&self.frame);
                return self.flush_pending();
            }
            // Not pending: either this touch was already settled and
            // relayed live (most touches), in which case this frame
            // carries real release events the compositor needs to see, or
            // there's nothing active and nothing to do either way.
            return self.relay_frame();
        }

        if self.touch_start.is_none() {
            // The first frame of a brand new touch.
            self.touch_start = Some(Instant::now());
            self.touch_max = count;
            self.settled = false;
            self.pending_frames.clear();
        } else {
            self.touch_max = self.touch_max.max(count);
        }

        if self.settled {
            // Already decided this touch is an ordinary gesture (or grew
            // past 3 into one) -- relay live from here on. The one
            // exception is newly reaching exactly 3 without ever having
            // lifted, well after the decision window closed: rare, but
            // still must not leak through as a real 3-finger touch on the
            // compositor. No debounce needed here -- growing an
            // already-settled touch all the way to a deliberate drag
            // without ever lifting first is rare enough not to warrant one.
            if count == 3 {
                self.enter_suppress()?;
                translator.mouse_down().await?;
                self.drive_drag(&active, translator).await?;
                return Ok(());
            }
            return self.relay_frame();
        }

        self.pending_frames.extend_from_slice(&self.frame);

        if self.touch_max >= 4 {
            // Unambiguously bigger than a 3-finger drag could ever be --
            // no need to wait out the rest of the window.
            self.settled = true;
            return self.flush_pending();
        }

        let elapsed = self.touch_start.unwrap().elapsed();

        if count == 1 && self.touch_max == 1 && elapsed >= translator.cfg.probe_delay {
            // Still just one finger after a short probe: ordinary pointer
            // movement, by far the most common case. Go live now rather
            // than waiting out the full entry_debounce, or every
            // touch-lift-reposition cycle of normal cursor use would add
            // a felt hitch.
            self.settled = true;
            return self.flush_pending();
        }

        if elapsed >= translator.cfg.entry_debounce {
            return self.resolve_touch_decision(translator, &active, count).await;
        }

        Ok(())
    }

    /// Called every poll tick (whether or not a new frame arrived) so a
    /// touch that's holding still (no new events at all) still gets its
    /// probe/confirm decision made on wall-clock time, not just on the
    /// next event.
    async fn resolve_touch_timeout(&mut self, translator: &mut GestureTranslator) -> Result<(), GtError> {
        if self.suppressing || self.settled {
            return Ok(());
        }
        let Some(since) = self.touch_start else { return Ok(()) };

        let active = self.active_slots();
        let count = active.len();

        if count == 1 && self.touch_max == 1 && since.elapsed() >= translator.cfg.probe_delay {
            self.settled = true;
            return self.flush_pending();
        }

        if since.elapsed() >= translator.cfg.entry_debounce {
            return self.resolve_touch_decision(translator, &active, count).await;
        }

        Ok(())
    }

    /// The entry_debounce window has closed: commit to a drag if the touch
    /// held stably at exactly 3 fingers the whole time, otherwise release
    /// it to the compositor as an ordinary gesture.
    async fn resolve_touch_decision(&mut self, translator: &mut GestureTranslator, active: &[usize], count: usize) -> Result<(), GtError> {
        if count == 3 && self.touch_max == 3 {
            self.pending_frames.clear();
            self.settled = true;
            self.enter_suppress()?;
            translator.mouse_down().await?;
            self.drive_drag(active, translator).await?;
            return Ok(());
        }
        self.settled = true;
        self.flush_pending()
    }

    /// Releases a buffered touch to the compositor: it either never
    /// reached 3 fingers, or grew past 3 into a bigger gesture that isn't
    /// ours to intercept. Replays the exact frames as they happened, then
    /// marks whatever is active now as relayed so later frames diff
    /// correctly.
    fn flush_pending(&mut self) -> Result<(), GtError> {
        if !self.pending_frames.is_empty() {
            self.synth.write(&self.pending_frames)?;
            trace!("Flushed a buffered touch ({} events) to the synthetic device.", self.pending_frames.len());
        }
        for slot in 0..MAX_SLOTS {
            self.relayed_active[slot] = self.slots[slot].tracking_id >= 0;
        }
        self.pending_frames.clear();
        Ok(())
    }

    fn enter_suppress(&mut self) -> Result<(), GtError> {
        self.suppressing = true;

        let mut release_frame = Vec::new();
        for slot in 0..MAX_SLOTS {
            if self.relayed_active[slot] {
                release_frame.push(abs_event(ABS_MT_SLOT, slot as i32));
                release_frame.push(abs_event(ABS_MT_TRACKING_ID, -1));
                self.relayed_active[slot] = false;
            }
        }
        if !release_frame.is_empty() {
            release_frame.push(syn_report());
            self.synth.write(&release_frame)?;
            trace!("Released all relayed slots to the synthetic device before suppressing.");
        }
        Ok(())
    }

    async fn drive_drag(&mut self, active: &[usize], translator: &mut GestureTranslator) -> Result<(), GtError> {
        let reference = match self.drag_ref_slot {
            Some(s) if active.contains(&s) => s,
            _ => {
                // first frame of the gesture, or our previous reference
                // finger lifted and a different one took its place:
                // re-baseline without applying a delta this frame.
                let s = active[0];
                self.drag_ref_slot = Some(s);
                self.drag_last_pos = Some((self.slots[s].x, self.slots[s].y));
                return Ok(());
            }
        };

        let (x, y) = (self.slots[reference].x, self.slots[reference].y);
        if let Some((lx, ly)) = self.drag_last_pos {
            let dx_mm = (x - lx) as f64 / self.x_res;
            let dy_mm = (y - ly) as f64 / self.y_res;
            translator.update_cursor_position(dx_mm * PX_PER_MM, dy_mm * PX_PER_MM).await?;
        }
        self.drag_last_pos = Some((x, y));
        Ok(())
    }

    fn relay_frame(&mut self) -> Result<(), GtError> {
        if self.frame.is_empty() {
            return Ok(());
        }
        for slot in 0..MAX_SLOTS {
            self.relayed_active[slot] = self.slots[slot].tracking_id >= 0;
        }
        self.synth.write(&self.frame)?;
        Ok(())
    }
}
