LifeSize Camera 10x
===================

[LifeSize](https://www.lifesize.com) builds conferencing equipment. Part of
what they build is dedicated hardware that uses high(ish)-quality HDMI cameras
for conference rooms.

The cameras - the "LifeSize Camera 10x" is an example - require 18V power,
and have an HDMI output. However, they also support PTZ and other
configuration. They can be obtained - with a bit of luck - really cheap.

This is done via EDID. The camera constantly polls the EDID from the host,
and looks for special "Chill"-packets. Information is sent back via HDMI in
SPD frames, though it's not necessary to receive them.

Without this control, the cameras are not truly useful - for example, Auto
Focus will be disabled by default, and the PTZ functionality is useless.

I've built a small setup where I can emulate an EDID using an I2C-target
microcontroller, and I can control the camera. I then use a cheap HDMI-to-USB dongle to receive the images.

Note that it is necesary to first emulate a good EDID and then switch to the
magic chill packets. A trace is contained in `chillboot`.

The commands are generated in a somewhat interesting way - the first and
second half of an ASCII string is "hashed" via CRC16, and the combination of
these halves is the command ID. I've extracted valid command IDs from the
firmware of the camera, but I haven't found the ascii representation of all
of them yet. Still, there's enough to operate Pan/Tilt/Zoom/Focus and set
HDMI resolution.

In theory it's also possible to update the firmware over this interface, so
be somewhat careful with sending arbitrary stuff. (In practice though I've
did this a lot and it did not cause harm. Also I have a backup of the
flash.)
