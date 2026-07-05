//! Diagnostic injector: emits a canonical, PERFECT 4-finger flick from
//! a fake touchpad directly at the compositor -- no proxy involved.
//! Used to determine whether gesture-completion problems live in the
//! proxy's relay or in the compositor's own gesture thresholds.
//!
//!     cargo test --test inject_flick -- --ignored --nocapture
//!
//! Watch the screen: does the overview/desktop-switch complete?

use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::thread::sleep;
use std::time::Duration;

use input_linux::{
    sys, AbsoluteAxis, AbsoluteInfo, AbsoluteInfoSetup, EventKind, InputId, InputProperty, Key,
    UInputHandle,
};

const EV_SYN: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const EV_ABS: u16 = 0x03;
const ABS_MT_SLOT: u16 = 0x2f;
const ABS_MT_TRACKING_ID: u16 = 0x39;
const ABS_MT_POSITION_X: u16 = 0x35;
const ABS_MT_POSITION_Y: u16 = 0x36;
const ABS_MT_TOUCH_MAJOR: u16 = 0x30;
const BTN_TOUCH: u16 = 0x14a;
const BTN_TOOL_FINGER: u16 = 0x145;
const BTN_TOOL_DOUBLETAP: u16 = 0x14d;
const BTN_TOOL_TRIPLETAP: u16 = 0x14e;
const BTN_TOOL_QUADTAP: u16 = 0x14f;

fn raw(t: u16, c: u16, v: i32) -> sys::input_event {
    let mut e: sys::input_event = unsafe { std::mem::zeroed() };
    e.type_ = t;
    e.code = c;
    e.value = v;
    e
}

fn pad() -> UInputHandle<File> {
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open("/dev/uinput")
        .expect("open /dev/uinput");
    let h = UInputHandle::new(f);
    h.set_evbit(EventKind::Key).unwrap();
    for k in [
        Key::ButtonTouch,
        Key::ButtonToolFinger,
        Key::ButtonToolDoubleTap,
        Key::ButtonToolTripleTap,
        Key::ButtonToolQuadtap,
        Key::ButtonLeft,
    ] {
        h.set_keybit(k).unwrap();
    }
    h.set_evbit(EventKind::Absolute).unwrap();
    let abs = |axis, min, max, res| AbsoluteInfoSetup {
        axis,
        info: AbsoluteInfo {
            value: 0,
            minimum: min,
            maximum: max,
            fuzz: 0,
            flat: 0,
            resolution: res,
        },
    };
    // mirror the bcm5974's real geometry (94/92 units per mm)
    let setups = [
        abs(AbsoluteAxis::X, -4750, 5280, 94),
        abs(AbsoluteAxis::Y, -150, 6730, 92),
        abs(AbsoluteAxis::MultitouchSlot, 0, 15, 0),
        abs(AbsoluteAxis::MultitouchTrackingId, 0, 65535, 0),
        abs(AbsoluteAxis::MultitouchPositionX, -4750, 5280, 94),
        abs(AbsoluteAxis::MultitouchPositionY, -150, 6730, 92),
        abs(AbsoluteAxis::MultitouchTouchMajor, 0, 2048, 0),
    ];
    for s in &setups {
        h.set_absbit(s.axis).unwrap();
    }
    h.set_propbit(InputProperty::Pointer).unwrap();
    h.set_propbit(InputProperty::ButtonPad).unwrap();
    let id = InputId {
        bustype: sys::BUS_USB,
        vendor: 0x3f3f,
        product: 0x0002,
        version: 1,
    };
    h.create(&id, b"3fd-flick-injector", 0, &setups).unwrap();
    sleep(Duration::from_millis(700)); // let the compositor adopt it
    h
}

fn frame(h: &UInputHandle<File>, evs: &[(u16, u16, i32)]) {
    let mut v: Vec<sys::input_event> = evs.iter().map(|&(t, c, va)| raw(t, c, va)).collect();
    v.push(raw(EV_SYN, 0, 0));
    h.write(&v).unwrap();
}

/// A textbook 4-finger flick: fingers land together, sweep fast over
/// `dy_units` in `steps` frames of 8ms, lift together.
fn flick(h: &UInputHandle<File>, dy_units: i32, steps: i32) {
    let xs = [-2000, -500, 1000, 2500];
    let y0 = 5800; // near the bottom, sweeping up
    let mut evs = Vec::new();
    for (slot, x) in xs.iter().enumerate() {
        evs.push((EV_ABS, ABS_MT_SLOT, slot as i32));
        evs.push((EV_ABS, ABS_MT_TRACKING_ID, 100 + slot as i32));
        evs.push((EV_ABS, ABS_MT_POSITION_X, *x));
        evs.push((EV_ABS, ABS_MT_POSITION_Y, y0));
        evs.push((EV_ABS, ABS_MT_TOUCH_MAJOR, 400));
    }
    evs.push((EV_KEY, BTN_TOUCH, 1));
    evs.push((EV_KEY, BTN_TOOL_QUADTAP, 1));
    frame(h, &evs);

    for step in 1..=steps {
        sleep(Duration::from_millis(8));
        let y = y0 - dy_units * step / steps;
        let mut evs = Vec::new();
        for slot in 0..4 {
            evs.push((EV_ABS, ABS_MT_SLOT, slot));
            evs.push((EV_ABS, ABS_MT_POSITION_Y, y));
        }
        frame(h, &evs);
    }

    sleep(Duration::from_millis(8));
    let mut evs = Vec::new();
    for slot in 0..4 {
        evs.push((EV_ABS, ABS_MT_SLOT, slot));
        evs.push((EV_ABS, ABS_MT_TRACKING_ID, -1));
    }
    evs.push((EV_KEY, BTN_TOUCH, 0));
    evs.push((EV_KEY, BTN_TOOL_QUADTAP, 0));
    frame(h, &evs);
}

/// Fingers land 0/25/60/120ms apart, already sweeping up at flick
/// speed the whole time -- the finger choreography the touch autopsies
/// measured on real fast vertical flicks.
fn staggered_flick(h: &UInputHandle<File>) {
    let xs = [-2000, -500, 1000, 2500];
    let land_at = [0i32, 3, 8, 15]; // in 8ms frames
    let y0 = 5800;
    let steps = 22; // ~176ms total
    let dy = 3200;
    let mut down = [false; 4];
    let tool_bits = [
        BTN_TOOL_FINGER,
        BTN_TOOL_DOUBLETAP,
        BTN_TOOL_TRIPLETAP,
        BTN_TOOL_QUADTAP,
    ];
    let mut prev_count = 0usize;
    for step in 0..steps {
        let y = y0 - dy * step / steps;
        let mut evs = Vec::new();
        for slot in 0..4usize {
            if land_at[slot] == step {
                evs.push((EV_ABS, ABS_MT_SLOT, slot as i32));
                evs.push((EV_ABS, ABS_MT_TRACKING_ID, 200 + slot as i32));
                evs.push((EV_ABS, ABS_MT_POSITION_X, xs[slot]));
                evs.push((EV_ABS, ABS_MT_POSITION_Y, y));
                evs.push((EV_ABS, ABS_MT_TOUCH_MAJOR, 400));
                down[slot] = true;
            } else if down[slot] {
                evs.push((EV_ABS, ABS_MT_SLOT, slot as i32));
                evs.push((EV_ABS, ABS_MT_POSITION_Y, y));
            }
        }
        // real firmware reports finger count via BTN_TOOL_* transitions
        let count = down.iter().filter(|d| **d).count();
        if count != prev_count {
            if prev_count > 0 {
                evs.push((EV_KEY, tool_bits[prev_count - 1], 0));
            }
            evs.push((EV_KEY, tool_bits[count - 1], 1));
            if prev_count == 0 {
                evs.push((EV_KEY, BTN_TOUCH, 1));
            }
            prev_count = count;
        }
        frame(h, &evs);
        sleep(Duration::from_millis(8));
    }
    let mut evs = Vec::new();
    for slot in 0..4 {
        evs.push((EV_ABS, ABS_MT_SLOT, slot));
        evs.push((EV_ABS, ABS_MT_TRACKING_ID, -1));
    }
    evs.push((EV_KEY, BTN_TOOL_QUADTAP, 0));
    evs.push((EV_KEY, BTN_TOUCH, 0));
    frame(h, &evs);
}

#[test]
#[ignore = "drives the live compositor; run manually (3FD_FLICK selects the variant)"]
fn inject_canonical_flicks() {
    let h = pad();
    let which = std::env::var("3FD_FLICK").unwrap_or_else(|_| "1".into());
    match which.as_str() {
        // fast flick -- 30mm up in ~100ms (typical human flick)
        "1" => flick(&h, 2760, 12),
        // faster, longer flick -- 45mm in ~90ms
        "2" => flick(&h, 4140, 11),
        // long deliberate swipe -- 55mm over 400ms
        "3" => flick(&h, 5060, 50),
        // very fast, short flick -- 20mm in ~70ms
        "4" => flick(&h, 1840, 8),
        // keep the device alive for external inspection
        "hold" => sleep(Duration::from_secs(25)),
        // wait for a proxy to attach, then run the fast flick through it
        "via-proxy" => {
            sleep(Duration::from_secs(6));
            flick(&h, 2760, 12); // same as variant 1
            sleep(Duration::from_secs(2));
        }
        // staggered, realistic flick: fingers land asynchronously while
        // already moving (mimics the journal's real-flick autopsies)
        "staggered" => {
            sleep(Duration::from_secs(6));
            staggered_flick(&h);
            sleep(Duration::from_secs(2));
        }
        _ => panic!("3FD_FLICK must be 1-4 or hold"),
    }
    sleep(Duration::from_millis(900)); // let the compositor settle
    h.dev_destroy().unwrap();
}
