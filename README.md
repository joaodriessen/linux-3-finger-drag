> [!IMPORTANT]
> **Superseded — please read before using this branch.**
>
> **On libinput ≥ 1.28 you probably don't need any of this.** libinput has
> native three-finger dragging built in; it's just disabled by default and
> most compositors don't expose a switch. Turning it on uses libinput's own
> (excellent) finger detection, keeps four-finger gestures fully native, and
> adds no duplicate device — far simpler and more robust than this proxy. I
> enable it with a small dependency-free `LD_PRELOAD` shim that works on any
> Wayland compositor (KDE, GNOME, …):
> **[enable-3fg-drag](https://github.com/joaodriessen/enable-3fg-drag)**.
> That's what runs on my own machine now — this proxy no longer does.
>
> **Where the proxy lives now:** the v2 evdev-proxy work was **merged
> upstream** into
> [lmr97/linux-3-finger-drag](https://github.com/lmr97/linux-3-finger-drag)
> (PRs #24–28). If you genuinely need the proxy — libinput < 1.28, or a setup
> where the native feature can't be enabled — get it from upstream rather
> than this fork.
>
> **What this branch is:** experimental work that came *after* the upstream
> stack — four-finger gesture scaling, velocity latching, liftoff glide,
> silent assembly, and clone-side drags. It is not upstream and is not
> maintained, but the docs below do describe what's actually on this branch.

# Three-Finger Drag for Linux (evdev-proxy fork)

Rest three fingers on the touchpad and move them: the window / text / icon
under the cursor is dragged, exactly like macOS's "three finger drag".
Lift, and the drag ends.

This is a fork of [lmr97/linux-3-finger-drag](https://github.com/lmr97/linux-3-finger-drag)
that replaces the original libinput-gesture-listener design with a full
**evdev multitouch proxy**. That change exists because of KWin: KDE
hardcodes desktop-switching to *both* 3- and 4-finger horizontal swipes,
with no setting to disable just the 3-finger binding — so a gesture
listener that merely *watches* the touchpad can never stop KWin from also
acting on the same three fingers. Owning the device and deciding
per-frame what the compositor gets to see is the only clean fix.

## How it works

```
 real touchpad ──(exclusive grab)──> linux-3-finger-drag ──> synthetic touchpad clone
                                            │                 · verbatim mirror of
                                        gesture               ·   everything that is
                                        machine               ·   NOT a 3-finger drag
                                                              · a 3-finger drag becomes
                                                              ·   ONE synthetic finger
                                                              ·   with BTN_LEFT held
```

There is exactly **one** output device. An earlier design also created a
virtual REL-mouse to carry drags; it was removed, because libinput runs
mice and touchpads through different pointer-acceleration curves, so a
mouse-driven drag could never feel like the cursor no matter how it was
scaled. Drags now happen *on the clone* as a synthetic finger with the
button held (the clone is a clickpad — button + moving finger is a drag
to libinput), so drag motion goes through the exact same touchpad
pipeline as ordinary cursor movement: same device, same curve, same
desktop speed setting. Identical feel by construction, and one system
knob tunes both.

* The real touchpad is **exclusively grabbed** for the program's whole
  lifetime. The compositor instead reads a **synthetic clone** that
  impersonates the real device's identity (name/vendor/product), so
  saved per-device settings (natural scrolling, tap-to-click, pointer
  accel…) keep applying. The clone carries a `phys` marker
  (`linux-3-finger-drag/proxy`) so the proxy can always tell its own
  clone apart from real hardware.
* A fresh touch is **withheld** from the compositor until classified:
  a lone finger goes live after `probeDelay` (default 15 ms — ordinary
  pointer motion never feels delayed), an ambiguous 2-3 finger touch
  waits out `entryDebounce` (default 50 ms), 4+ fingers goes live
  instantly. Real fingers land and lift asynchronously; judging a touch
  frame-by-frame (the naive approach) leaks phantom taps and misreads
  gestures.
* A touch that holds at **exactly 3 fingers** through the debounce
  becomes a drag: the compositor never learns those fingers existed
  (KWin can't desktop-switch on what it can't see). The real fingers'
  motion is replayed as one synthetic finger on the clone with
  `BTN_LEFT` held. Near a pad edge that finger **hops** — it lifts and
  re-lands at pad centre while the button stays down — so a long drag
  never runs out of physical pad. The drag ends when the last finger
  lifts; staggered liftoffs can't leak trailing 1-2 finger touches
  (which libinput would read as a right-click tap).
* Anything else — taps (3-finger tap still middle-clicks!), scrolls,
  4-finger gestures, quick flicks — is relayed to the clone, verbatim
  by default. Four-plus-finger touches can optionally be **scaled**;
  see [Four-finger gestures and flicks](#four-finger-gestures-and-flicks).

The classification logic lives in a pure, I/O-free state machine
(`src/runtime/gesture.rs`) driven by an injected clock, with a
regression test suite encoding every failure mode this project has hit
live (`src/runtime/gesture/tests.rs`). The event loop is fully
event-driven (epoll on the device fd + exact decision deadlines): idle
CPU is zero, and no polling interval sits between your fingers and a
decision.

## Four-finger gestures and flicks

Owning the touchpad means 4-finger gestures reach the compositor only if
the proxy hands them over faithfully. Two things make that harder than a
straight copy, and both are handled here.

**Scaling.** KWin's 4-finger gestures have no sensitivity setting of
their own. `fourFingerScale` (default `1.0` = untouched passthrough)
scales 4+ finger motion as relayed to the compositor, so you can slow
those gestures down without touching cursor speed, scrolling, or drags.

**Flicks.** A fast flick is exactly where naive relaying breaks:

* Fingers on a real pad land **staggered** — the faster the hand, the
  bigger the stagger. A brief 1- or 2-finger apparition that then grows
  and vanishes poisons libinput's gesture recognition. So a touch moving
  faster than a threshold while still gaining fingers is **withheld**
  ("silent assembly", up to 160 ms) and then introduced as one clean
  simultaneous multi-finger touch.
* Scaling a flick down starves it: KWin cancels swipes whose accumulated
  distance misses its completion threshold. So flick speed is measured
  in **pad-lengths per second, per axis** (fair to both directions on a
  pad that's wider than it is tall, and resolution-independent across
  hardware); once a touch reaches full flick speed the scale **latches**
  at unscaled for the rest of that touch — once flicked, committed.
* Motion withheld during assembly is not discarded but repaid as
  **debt** during live relay, so a recognised flick delivers all of its
  physical travel, and a modest boost is applied on top so gestures
  complete decisively instead of hovering at the compositor's threshold.
* On liftoff a flicked gesture **glides**: the synthetic fingers coast
  along the launch vector with decay for up to 300 ms, the way inertia
  works on macOS, then release cleanly. A new real touch ends a glide
  instantly.

These are internal constants in `src/runtime/gesture.rs` (`FLICK_*`,
`FAST_ASSEMBLY_*`, `GLIDE_*`, `EASE_IN_*`), tuned on the hardware below
rather than exposed as config. `fourFingerScale` is the one knob.

## Requirements

* Rust toolchain (build-time only — there are **no** C library
  dependencies; the program speaks evdev/uinput directly)
* `uinput` kernel module
* read access to `/dev/input` (user in the `input` group) and write
  access to `/dev/uinput` (udev rule included)
* a systemd user session for the provided unit (any init works if you
  start the binary yourself)

Wayland and X11 are both fine; the proxy operates below the display
server. Developed and tuned on a MacBookPro11,3 (bcm5974 touchpad)
running CachyOS + KDE Plasma Wayland.

## Installation

Automated (installs udev rule, adds you to `input`, builds, installs
binary + config + systemd user unit):

```bash
sudo ./install.sh
```

Manual:

```bash
# 1. permissions
sudo cp 60-uinput.rules /etc/udev/rules.d/
sudo gpasswd --add $USER input
echo uinput | sudo tee /etc/modules-load.d/uinput.conf
sudo modprobe uinput
# log out & in (or reboot) so the group change applies

# 2. build & install
cargo build --release
sudo cp target/release/linux-3-finger-drag /usr/bin/

# 3. config + service
mkdir -p ~/.config/linux-3-finger-drag
cp 3fd-config.json ~/.config/linux-3-finger-drag/
mkdir -p ~/.config/systemd/user
cp three-finger-drag.service ~/.config/systemd/user/
systemctl --user enable --now three-finger-drag.service
```

Test in the foreground first if you're changing code:
`./target/release/linux-3-finger-drag` (Ctrl-C to quit — the touchpad
returns to normal the moment the process exits).

### CLI

```
linux-3-finger-drag [--device /dev/input/eventN]
```

`--device` skips touchpad auto-discovery and proxies the given device.
Used by the integration test harness; also handy on machines with more
than one touchpad (auto-discovery proxies the first one found).

## Configuration

`~/.config/linux-3-finger-drag/3fd-config.json`, hot-reloaded on change
(log settings excepted — those need a restart). All fields optional:

| field | default | meaning |
|---|---|---|
| `acceleration` | `1.0` | drag motion relative to ordinary cursor motion. `1.0` means a drag moves exactly as far as your finger would move the cursor — the drag rides the same touchpad pipeline, so your desktop's pointer-speed setting tunes both together. Use this only to make dragging deliberately faster (`> 1`) or slower (`< 1`) *than* the cursor |
| `dragEndDelay` | `0` | drag-lock, in ms: after lifting, the button stays held this long, and a new 3-finger touch inside the window **continues the same drag**. Any other touch releases the button *before* it is relayed, so post-drag pointer motion can never smear the held button around. `0` disables. |
| `entryDebounce` | `50` | ms an ambiguous (2-3 finger, possibly still growing) fresh touch is withheld before committing: drag, or replay to the compositor |
| `probeDelay` | `15` | ms a so-far-lone finger is withheld (just long enough to catch a 2nd/3rd finger landing a beat behind the 1st) |
| `pressGrace` | `75` | ms a committed drag defers its button press while the fingers haven't moved. Lets a 4th finger that lands *after* the entry window (fast, sloppy 4-finger swipes stagger hard) abort the misclassified drag with no phantom click — the touch is handed to the compositor mid-gesture instead |
| `fourFingerScale` | `1.0` | motion scale for 4+ finger touches as relayed to the compositor. `1.0` = verbatim passthrough; lower values slow KWin's 4-finger gestures without affecting the cursor, scrolling, or drags. Recognised flicks override this (see above) |
| `logFile` | `"stdout"` | log destination (`"stdout"` or a file path) |
| `logLevel` | `"info"` | `off` / `error` / `warn` / `info` / `debug` / `trace` |

The old `responseTime` knob is gone: the loop is event-driven, so there
is no poll interval to tune. A leftover `responseTime` in an existing
config file is ignored harmlessly.

## Testing

```bash
cargo test                                                   # gesture suite + fuzzer (pure, instant)
cargo test --test integration -- --ignored --test-threads=1  # software-in-the-loop, see below
```

`cargo test` is pure and hermetic: 43 regression tests over the state
machine — each one encoding a failure this project actually hit on real
fingers — plus an invariant fuzzer. No devices, no timing flake.

The integration test creates a **fake touchpad** via uinput, points the
real compiled binary at it (`--device`), injects scripted multi-finger
sequences, and asserts on what actually comes out of the clone
(synthetic drag finger, button state, no real-finger leaks). It covers
the whole evdev path without involving your real touchpad.

> [!CAUTION]
> Its devices are **real input devices** — your compositor will act on
> them. The test parks the cursor against the right screen edge and the
> drag scenario holds the left button there for ~100 ms. Run it only
> from a session where a stray click at the right screen edge is
> harmless. This is why it's `#[ignore]`d by default.

`tests/inject_flick.rs` is a manual diagnostic (also `#[ignore]`d) that
injects choreography-faithful flicks — including the staggered
finger-landing pattern real hardware produces — for debugging gesture
recognition end to end.

## Troubleshooting

* **Touchpad dead while the program runs?** The proxy has the device
  grabbed but something is failing after that. Check
  `journalctl --user -u three-finger-drag.service -e` — and note the
  touchpad always returns the instant the process exits.
* **"You are not yet allowed to write to /dev/uinput"** — udev rule not
  applied, or you haven't logged out and back in since being added to
  the `input` group.
* **Drag feels too slow/fast** — first check your desktop's ordinary
  pointer-speed setting: drags ride the same pipeline as the cursor, so
  that one setting moves both. Only reach for `acceleration` if you want
  dragging to differ *from* cursor speed.
* **KDE gestures still firing on 3 fingers?** Then the compositor is
  reading the *real* touchpad, not the clone — the service probably
  isn't running.
* **4-finger gestures feel too fast / too sensitive?** Lower
  `fourFingerScale` (e.g. `0.3`). Cursor, scrolling and drags are
  unaffected.
* **Fast 4-finger flicks don't complete the gesture?** That's the case
  the flick machinery exists for; if it regresses, run with
  `"logLevel": "debug"` and look for the per-touch autopsy lines
  (measured velocity in pads/s, whether the flick latched), then
  reproduce it headlessly with `tests/inject_flick.rs`.
* **Two blank/duplicate touchpads in your settings panel?** That's the
  clone showing up alongside the real device. Expected with this design
  — and one of the reasons native libinput 3-finger drag (see the banner
  at the top) is the better route where it's available.
* **Two touchpads?** Auto-discovery takes the first; pin one explicitly
  with `--device`.

## License

MIT (see `LICENSE`). Based on
[lmr97/linux-3-finger-drag](https://github.com/lmr97/linux-3-finger-drag);
the evdev-proxy architecture, gesture state machine, and test harness
are this fork's additions.
