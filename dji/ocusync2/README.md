DJI OcuSync 2.0 RF investigations
=================================

DJI OcuSync 2.0 is an OFDM-based "custom" communication scheme. DJI
describes it as "SDR", probably because it's implemented in software
running on a custom ASIC ("Sparrow" aka "S1", or "Pigeon").

It is not based on WiFi or other off-the-shelf communication systems, but it
shares a few properties with LTE. Likely someone read the LTE PHY spec and
used it as "good practices". Similarity stops quickly so there's no point in
attempting to use anything LTE specific.

Packets can be 20MHz, 10MHz, 3MHz, 1.4MHz ([FCC filing](https://fccid.io/SS3-MT2PD2007/Test-Report/Test-Report-SRD-4941236.pdf)),
and there's a 1.4MHz-CA (likely "collision avoidance" aka. frequency
hopping, which is used for the uplink). 

OFDM modulation is used for all packets; the occupied BW is slighly lower
than the gross BW, for example 20MHz uses 1201 subcarriers with 15kHz each, 
so ~18MHz OBW.

Symbol duration is 15kHz, plus cyclic prefix (CP). By oversampling, one gets
to a power-of-two length per sample, so for example 20MHz can be oversampled
to 30.72MHz (=2048 * 15000kHz). Due to the CP, the net symbol rate is
slighly lower than 15kHz. With an FFT size of 2048, only the middle 1201
subcarriers are used (for 20 MHz).

CP length is 144 (for 20MHz), 72 (for 10MHz) etc.; some symbols, notably the
first, last and a middle one for regular DL packets (the symbol structure 
depends on the data type), have an longer CP (144+16 for 20 MHz, 72+8 for 10
MHz).

Packets can be downlink (drone-to-RC), uplink (RC-to-drone), and broadcast
(remote drone id aka. flight info). They all use OFDM, but with different
bandwidths, and the uplink is using frequency-hopping.

Each symbol is either a reference or a data symbol. For regular downlink
packets, the structure is:

   RS0 | D | D | D | D | D | D | RS1 | D | D | D | D | D | D | RS0

(So 15 symbols total, Packet length is ~1ms)

Reference symbols are LTE-like [Zadoff-Chu sequences](https://en.wikipedia.org/wiki/Zadoff%E2%80%93Chu_sequence),
with the DC carrier not removed but set to a constant. ZC sequences are
generated in frequency domain, with a sequence length of 1201 (20MHz) or 601
(10MHz), and various sequences.

Data symbols are either QPSK or higher-order QAM. The first signal is always
QPSK. The DC subcarrier (for example 600 for 20 MHz) is fixed and not used
for data. There are no obvious pilots (or I haven't found them).
