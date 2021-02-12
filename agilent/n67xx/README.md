Agilent/Keysight N67xx-Series
=============================

The Agilent/Keysight N67xx-series DC power supplies are somewhat interesting
because they are modular. There's a "mainframe" (N6700A/B/C, N6701B/C,
N6702B/C, N6705A/B/C) that supplies a bulk 48V and provides a user
interface, and then there are various modules that can be plugged in.

There are a few categories of modules, from basic (N673xB/N674xB) to more
advanced (higher-precision, higher-range) ones (N676x), or even SMUs
(N678x) and e-Loads (N679x).

Communication to the module is entirely digital, though with a somewhat
interesting scheme. Individual modules are galvanically isolated, using a
transformer for power transmission and optocoupler for the digital part.
Additionally, a 12V is included in the module that is powered directly from
the mainframe.

Mainframes exist as either 19" racks (N6700/01/02), or as bench units
(N6705). They are either PowerPC-based, running VxWorks (A, B models),
or ARM-based, running WinCE (C models). 

Each module contains (at least) two PCBs, a large power PCB and a small
control PCB that is plugged into a 168-pin DIMM socket. The control PCB
contains ADCs, DACs and an FPGA, and configures things like the voltage
and current setpoint, power limit setpoint, digitizes the current and
voltage, and handles host communication. Configuration resistors on the
power PCB allow to coarsely identify the module type, and then an I2C EEPROM
is present on the power PCB that contains calibration information.

The [specifications](https://literature.cdn.keysight.com/litweb/pdf/N6700-90001.pdf) give a good overview of the abilities of the different modules. All
digitizing happens inside the modules, either using Sigma-Delta-ADCs or real
ADCs (take a look at "measurement resolution", that typically gives away
whether real ADCs are present or not). Basic modules contain a single ADC
(which is muxed to the different measurement sources), more advanced modules
can measure two things at the same time.

Communication is done over SPI after initial configuration at a fixed rate. 
Each communication frame is 64-bit and can contain a measurement, a command
response (from the device), or a command (from the host).

One a module is configured to a specific current/voltage setpoint, it will
regulate to these settings, including CC, CV modes, over-voltage and
-current protection etc.; status and measurements can be queried by the
host, and also regular-interval measurements are automatically streamed.

This means that even the most basic modules allow for kHz-range digitization
of measurements.

Arbitrary waveform generation is also possible, though bandwidth is
individually limited by slew time. The most advanced modules (N678x) can
generate signals in the kHz-range, the basic modules are limited to a few
hundred Hz. For ARB, simply flood the device with "set voltage" commands.

The modules can also store lists and execute them without host intervention,
though list size is limited of course.

Initial configuration is done by talking JTAG to the FPGA. A special
sequence first queries the ID code from the module, then configures a
bitstream particular for this FPGA model, then queries configuration
resistors over boundary scan, and then configures the final bitstream. After
that, SPI communication is started, the EEPROM is read, and - with the
calibration constants from the EEPROM - the operational mode can be
configured.