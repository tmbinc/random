#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import scipy.signal as signal
from test import zcsequence

# some tools

def corr(x, y=None):
    if y is None:
        y = x
    result = np.correlate(x, y, mode='full')
    return result[result.size//2:]

def fshift(y, offset, Fs):
	x = np.linspace(0.0, len(y)/Fs, len(y))
	return y * np.exp(x * 2j * np.pi * offset)

class SpectrumCapture:
	def __init__(self, fn, dtype = ">f"):
		self.y = np.fromfile(fn, dtype=dtype).astype(np.float32).view(np.complex64)
		self.Fs = 200e6
		self.pkt_starts = []

	def coarse_collect_packets(self, level, max = None, debug = False, start = 0, pkt_time = 1.2e-3):
		"""Simple squelch-based attempt to find start of packets"""
		pos = int(start * self.Fs)
		yabs = np.abs(self.y)
		if debug:
			plt.plot(yabs[::100])
			plt.hlines([level], xmin=0, xmax=len(yabs)//100)
			plt.show()
		above_level = yabs >= level
		while pos < len(yabs):
			if max is not None and len(self.pkt_starts) >= max:
				break
			start = np.argmax(above_level[pos:]) + pos
			print("pkt end at", pos, "next start at", start)
			if not yabs[start] >= level:
				break

			self.pkt_starts.append(start)
		
			pos = start + int(pkt_time * self.Fs)

	def obtain_packet_raw(self, pktnum, before, after, Fcarrier, Fsnew = 30.72e6):
		pkt_fullrate = self.y[self.pkt_starts[pktnum] - int(before*self.Fs):self.pkt_starts[pktnum] + int(after*self.Fs)]
		# shift by ~carrier
		x = np.linspace(0.0, len(pkt_fullrate)/self.Fs, len(pkt_fullrate))
		pkt_fullrate = pkt_fullrate *  np.exp(x * 2j * np.pi * -Fcarrier)

		# decimate
		y = np.interp(np.arange(0, len(pkt_fullrate), self.Fs/Fsnew), np.arange(0, len(pkt_fullrate)), pkt_fullrate)
		#y = signal.resample_poly(pkt_fullrate, 96, 625)
		return y

	def pkt(self, pktnum, Fcarrier, BW = 20, length = 1.1e-3):
		if BW == 20:
			Fs = 30.72e6
		elif BW == 10:
			Fs = 15.36e6
		else:
			assert False, "unsupported BW"

		return Packet(self.obtain_packet_raw(pktnum, .1e-3, length, Fcarrier, Fs), Fs, BW, pktnum = pktnum)

	def plot_spectrum(self):
		plt.specgram(self.y, Fs = self.Fs)
		plt.vlines([np.array(self.pkt_starts) / self.Fs], ymin=-self.Fs/2, ymax=self.Fs/2)
		plt.show()

class Packet:
	def __init__(self, y, Fs, BW, pktnum):
		self.Fs = Fs
		self.y = y
		self.start = 0
		self.pktnum = pktnum
		# agc
		y /= np.max(np.abs(y))
		if BW == 20:
			self.Ncarriers = 1201
			self.Nfft = 2048
			self.cp_length = [
				144 + 16, # 0
				144, # 1
				144, # 2
				144, # 3
				144, # 4
				144, # 5
				144, # 6
				144, # 7
				144 + 16, # 8
				144, # 9
				144, # 10
				144, # 11
				144, # 12
				144, # 13
				144 + 16, # 14
			]
			self.zc_sym = [0, 7, 14]
		elif BW == 10:
			self.Ncarriers = 601
			self.Nfft = 1024
			#array([-81, -69, -72, -81, -63, -73, -76, -68])
			self.cp_length = [
				72 + 8, # 0
				72, # 1
				72, # 2
				72, # 3
				72, # 4
				72, # 5
				72, # 6
				72, # 7
				72, # 8
			]
			self.zc_sym = [3, 5]
			

		#self.plot_spectrum(real = True)

		self.find_fine_start(False)

	def find_fine_start(self, debug = False):

		print(np.mean(np.abs(self.y[4000:4100])))
		if np.mean(np.abs(self.y[4000:4100])) < .05:
			print("skipping prefix!")
			self.y = self.y[4100:]

		res = []
		cpl = self.cp_length[0]
		for n in range(self.Nfft, len(self.y) - cpl):
			ac = np.sum(self.y[n:n+cpl] * np.conj(self.y[n-self.Nfft:n-self.Nfft+cpl]))
			res.append(ac)

		if debug:
			# get a rough idea of the first peak
			x = np.linspace(0, len(self.y) / (self.Nfft + cpl) , len(res))
			plt.plot(x, np.array(res)*300) # corr(self.y, self.y))
			plt.specgram(self.y, Fs=self.Nfft + cpl, NFFT=self.Nfft//64, window = matplotlib.mlab.window_none, noverlap=0)
#		plt.plot(res)
			plt.show()

		first_symbol_start = np.nonzero(np.abs(res) > 5.0)[0][0]
		print("fss", first_symbol_start)

		if debug:
			# show the correlation as absolute values
			plt.plot(np.abs(res))
			plt.show()

		# fine-tune
		search_range = 400
		if first_symbol_start < search_range:
			first_symbol_start = search_range

		if debug:
			# show that the search range contains the peak
			plt.plot(res[first_symbol_start-search_range:first_symbol_start+search_range])
			plt.show()

		first_symbol_start = np.argmax(res[first_symbol_start-search_range:first_symbol_start+search_range])+first_symbol_start-search_range
		print("fine-tuned first symbol start", first_symbol_start)

		self.start = first_symbol_start
		self.detected_ffo = self.Fs / (2 * np.pi * self.Nfft) * np.angle(res[first_symbol_start])

	def sym_offset(self, s):
		sym_start = self.start + s * self.Nfft + int(np.sum(self.cp_length[:s]))
		sym_end = self.start + (s + 1) * self.Nfft + int(np.sum(self.cp_length[:s+1]))
		return sym_start, sym_end

	def sym(self, s, skip_cp = False, shift = 0):
		sym_start, sym_end = self.sym_offset(s)
		sym_start += shift
		sym_end += shift
		if skip_cp:
			sym_start += self.cp_length[s]
		return self.y[sym_start:sym_end].copy()

	def find_zc_seq(self, s, debug = False):
		res = []
		for r in range(1, self.Ncarriers-1):
			a = zcsequence(r, self.Ncarriers)
			sy = self.sym(s, True)
			fft_tr = self.tfft(sy)
			res.append(np.max(np.abs(corr(fft_tr, a))))

		best = np.argmax(res) + 1
		if debug:
			print("best zc seq", best)

			# plot the correlation; there should be ONE SINGLE hit,
			# otherwise things are terribly wrong.
			plt.plot(res)
			plt.show()

		return best

	def find_zc_offset(self, s, seq, cyc):
		oy = self.y.copy()
		a = zcsequence(seq, self.Ncarriers, cyc)

		resx = []
		resy = []

		for i in np.linspace(-5, 5, 1000):
			self.y = oy.copy()
			self.apply_sample_offset(i)
			sy = self.sym(s, True)

			#print("carrier 600 is", np.angle(sa[600]) / (2 * np.pi) * 360)
			#print("carrier 600 is", np.abs(sa[600]) / np.mean(np.abs(sa)))

			sa = self.tfft(sy)
			if (sa == 0).any():
				sa += 1
			adiff = np.angle(a / sa)
			adiff[self.Ncarriers//2] = adiff[self.Ncarriers//2+1]
			adiff = np.unwrap(adiff)

			#plt.plot(adiff)
			#plt.show()

			slope = np.max(adiff) - np.min(adiff)
			slope = (adiff - np.mean(adiff))
			slope = np.sqrt(np.mean(slope**2))
			resx.append(i)
			resy.append(slope)

		self.y = oy
		print("minrms", np.min(resy), "at", resx[np.argmin(resy)])

		plt.plot(resx, resy)
		plt.show()
		return resx[np.argmin(resy)]

	def find_zc_shift(self, s, seq, cyc = 0, debug = False):
		a = zcsequence(seq, self.Ncarriers, cyc)
		sy = self.sym(s, True)
		am = np.argmax(np.abs(corr(self.tfft(sy), a)))

		real_cyc = (cyc - am) % (self.Ncarriers)

		if debug:
			#plt.plot(np.abs(corr(self.tfft(sy), a)))
			#plt.show()

			for cyc_offset in [0]:
				sample_offset = self.find_zc_offset(s, seq, (real_cyc + cyc_offset * 51) % self.Ncarriers)

			print("final sample offset", sample_offset)
			self.apply_sample_offset(sample_offset)

			sy = self.sym(s, True)
			sa = self.tfft(sy)
			adiff = np.angle(a / sa)
			adiff[self.Ncarriers//2] = adiff[self.Ncarriers//2+1]
			adiff = np.unwrap(adiff)
			print("phase shift remaining between", np.max(adiff), np.min(adiff))

			slope = np.max(adiff) - np.min(adiff)
			slope = (adiff - np.mean(adiff))
			slope = np.sqrt(np.mean(slope**2))
			print("slope", slope)

			print("phase 0", np.angle(sa[self.Ncarriers//2]))
			plt.plot(adiff)
			plt.show()

			return (real_cyc + cyc_offset * 51) % self.Ncarriers

		return real_cyc

	def find_zc(self, s = 0, zc_seq = None, zc_cyc = None, debug = False):
		if zc_seq is None:
			zc_seq = self.find_zc_seq(s, debug = debug)
		if zc_cyc is None:
			zc_cyc = self.find_zc_shift(s, zc_seq, debug = debug)
			if debug:
				print(f"detected zq_cyc {zc_cyc}")
		self.zc_seq = zc_seq
		self.zc_cyc = zc_cyc

	def plot(self):
		plt.plot(self.y)
		for s in range(len(self.cp_length)):
			s, _ = self.sym_offset(s)
			plt.vlines([s], ymin=-1, ymax=1)
			
		plt.show()

	def plot_spectrum(self, real = False, sym = None):
		if not sym:
			if not real:
				plt.specgram(self.y, Fs=self.Nfft + self.cp_length[1], NFFT=self.Nfft//64, window = matplotlib.mlab.window_none, noverlap=0)
			else:
				plt.specgram(self.y, Fs=self.Fs)
		else:
				plt.specgram(self.sym(sym), Fs=self.Nfft + self.cp_length[sym], NFFT=self.Nfft//64, window = matplotlib.mlab.window_none, noverlap=0)

		plt.show()

	def show_symbol(self, ax, s, sy, chan_est = None):
		fft = self.tfft(sy[self.cp_length[s]:self.cp_length[s]+self.Nfft])
		print(fft.shape)
		if chan_est is not None:
			fft *= chan_est
		x = [np.real(fft), np.imag(fft)]

		for i in range(6):
			ax.scatter(x[0][i*100:(i+1)*100], x[1][i*100:(i+1)*100], s=1)

	def show_all_symbol(self):
		fig,ax = plt.subplots(4,4)
		for s in range(len(self.cp_length)):
			self.show_symbol(ax[s//4][s%4], s, self.sym(s), chan_est)
		plt.show()

	def plot_all_symbol_spectrum(self):
		"""Simple overlay plot, to check for correct occupied bandwidth / IFO"""
		for s in range(len(self.cp_length)):
			if s == 14:
				continue
			fft = np.fft.fft(self.sym(s, True))
			plt.plot(np.abs(fft))
		plt.show()

	def show_cp(self):
		"""Shows the CP overlap for each symbol (slightly shfited upwards for better visibility)"""
		for s in range(len(self.cp_length)):
			sy = np.abs(self.sym(s))
			cp = self.cp_length[s]
			plt.plot(np.arange(0, len(sy)), sy + s)
			plt.plot(np.arange(len(sy) - cp, len(sy) - cp + len(sy)), sy + s + .01)
		plt.show()
		exit()

	def interactive(self):
		# interactive version

		fig,ax = plt.subplots(4,4)
		from matplotlib.widgets import Slider, Button
		pc = 0
		linrot = Slider(plt.axes([0.25, .1, 0.50, 0.02]), 'lin rot', pc - .01, pc + .01, valinit=pc)
		off = Slider(plt.axes([0.25, .08, 0.50, 0.02]), 'offset', -10, 10, valinit=0)
		tune = Slider(plt.axes([0.25, .06, 0.50, 0.02]), 'tune', -15000,15000, valinit=0)
		sr = Slider(plt.axes([0.25, .04, 0.50, 0.02]), 'sr', -.001,.001, valinit=0)
		seqcor = Slider(plt.axes([0.25, .02, 0.30, 0.02]), 'seq', -10,10, valinit=0)

		write = Button(plt.axes([0.60, .02, 0.20, 0.02]), 'write')

		def save(_):
			print("saving data...")
			for i, j in enumerate(self.symbol_data):
				if j is not None:
					with open("pkt_%d_sym_%d.txt" % (self.pktnum, i), "w") as fo:
						for i in j:
							fo.write("%f %f\n" % (np.real(i), np.imag(i)))

		write.on_clicked(save)

		def update(_):
			for a in ax:
				for a in a:
					a.clear()
			self.symbol_data = []
			for s in range(len(self.cp_length)):
				sy = self.sym(s)
				sy = fshift(sy, tune.val, self.Fs)
				sy = np.interp(np.arange(off.val, off.val+len(sy), 1+sr.val), np.arange(0, len(sy)), sy)
				fft = self.tfft(sy[self.cp_length[s]:self.cp_length[s]+self.Nfft])
				x = np.linspace(-.5 * linrot.val * len(fft), .5 * linrot.val * len(fft), len(fft))

				print(f"linrot {linrot.val} offset {off.val} tune {tune.val} sr {sr.val} seqcor {seqcor.val}")

	
				a = ax[s//4][s%4]
	
				if s in self.zc_sym:

					if False:
						seq = zcsequence(self.zc_seq, self.Ncarriers, self.zc_cyc + int(seqcor.val) * 51)
						est = seq / fft
						est[self.Ncarriers//2] = (est[self.Ncarriers//2-1] + est[self.Ncarriers//2+1]) * .5 # fake
						self.magest = np.abs(est)
						a.plot(np.angle(est))
					else:
						a.plot(np.abs(fft))
					self.symbol_data.append(None)
				else:
					fft *= np.exp(x * 2j * np.pi)
#					fft *= self.magest
#					fft /= fft[self.Ncarriers//2+1]
#					print(fft[self.Ncarriers//2])
					x = [np.real(fft), np.imag(fft)]
					self.symbol_data.append(fft)
					for i in range(self.Ncarriers//100):
						a.scatter(x[0][i*100:100*(i+1)], x[1][i*100:100*(i+1)], s=10/(i+10))
					pilots = [self.Ncarriers//2]
					a.scatter(x[0][pilots], x[1][pilots], s=5, marker='X')
					#i = np.arange(500, 700)
					#a.scatter(x[0][i], x[1][i], s=1)
			fig.canvas.draw_idle()
			#a.plot(np.abs(fft))
		linrot.on_changed(update)
		off.on_changed(update)
		tune.on_changed(update)
		sr.on_changed(update)
		seqcor.on_changed(update)
		update(0)

		plt.show()

	def time_domain_zc_corr(self, s, zc_seq, zc_cyc):
		# create time domain ZC
		# self.y *= np.exp(-1j * 2.1467642031119)

		print("Sym%d, using ZC %d:%d for time domain" % (s, zc_seq, zc_cyc))
		seq = zcsequence(zc_seq, self.Ncarriers, zc_cyc)

		# set c0, likely wrong
		seq[self.Ncarriers//2] = .9*2.4256186683261327j
		td = self.itfft(seq)
		# prefix with CP
		td = np.concatenate((td[-self.cp_length[s]:], td))

#		plt.plot(np.abs(td) / np.max(np.abs(td)))
#		plt.plot(np.abs(self.sym(s)) / np.max(np.abs(self.sym(s))))
		sa = self.sym(s)
		sa /= np.mean(np.abs(sa))
		td /= np.mean(np.abs(td))


		c = corr(sa, td)
		ref_phase = np.angle(c[0])
		print("Sy%d ref" % s, ref_phase)

		sa *= np.exp(-1j * ref_phase)

		print("sa", len(sa))
		print("td", len(td))

		plt.plot(np.real(sa))
		plt.plot(np.real(td))
		plt.show()

		c = corr(sa, td)
		print("ref angle after correction", np.angle(c[0]))

		plt.plot(np.abs(c))

#		plt.plot(np.unwrap(np.angle(self.tfft(td))))
#		plt.plot(np.unwrap(np.angle(self.tfft(sy_shifted))))
#		plt.plot(np.angle(self.tfft(td)))
#		plt.plot(np.angle(self.tfft(sy_shifted)))

#		plt.plot(np.arange(0, len(sy), 1), np.abs(sy))
#		plt.plot(np.arange(i_shift, i_shift + len(td), 1), np.abs(td))
		plt.show()

		sym = self.y

		co = np.correlate(sym, td, mode='valid')

		## prioritize things closer
		##co += np.abs(np.arange(0, len(co)) - self.sym_offset(sym)[0])

		off = np.argmax(np.abs(co))
		print("offset Sy%d = %d" % (s, off))

		plt.plot(np.abs(co) / 100.0)
		plt.plot(np.arange(off, off + len(td)), np.real(td))
		plt.plot(np.real(sym))
		plt.show()

	def time_domain_zc_corr_x(self, s, zc_seq, zc_cyc):
		# create time domain ZC
		print("using ZC %d:%d for time domain" % (zc_seq, zc_cyc))
		seq = zcsequence(zc_seq, self.Ncarriers, zc_cyc)
		seq[self.Ncarriers//2] = .9*2.4256186683261327j
		td = self.itfft(seq)
		td = np.concatenate((td[-self.cp_length[s]:], td))

		sa = self.sym(s)

		# adjust amplitude
		sa /= np.mean(np.abs(sa))
		td /= np.mean(np.abs(td))

		co = np.correlate(self.y, td, mode='valid')

		# find definitive first start
		self.start = np.argmax(np.abs(co))
		print("integer shift", self.start)
		ref_phase = np.angle(co[self.start])
		print("fixing up ref phase", ref_phase)
		self.y *= np.exp(-1j * ref_phase)

		self.magest = np.abs(self.tfft(td) / self.tfft(sa))

	def tfft(self, sy):
		fft = np.fft.fft(sy)
		half_carriers = self.Ncarriers//2
		return np.concatenate((fft[-half_carriers:], fft[:half_carriers+1]))

	def itfft(self, c):
		half_carriers = self.Ncarriers//2
		c_full = np.zeros((self.Nfft), dtype=np.complex64)
		c_full[-half_carriers:] = c[:half_carriers]
		c_full[:half_carriers+1] = c[half_carriers:]

		return np.fft.ifft(c_full)

	def with_sample_offset(self, offset):
		return np.interp(np.arange(offset, offset+len(self.y), 1), np.arange(0, len(self.y)), self.y)

	def apply_sample_offset(self, offset):
		self.y = self.with_sample_offset(offset)

	def linrot_time_domain_corr(self, s, linrot = -0.003033557046979867):
		
		# grab symbol, calc FFT, de-rotate with manual constant
		sy = self.sym(s, True)
		fft = self.tfft(sy)
		fft *= np.exp(2j * np.pi * np.linspace(-.5 * linrot * len(fft), .5 * linrot * len(fft), len(fft)))

		# extract mean magnitude so we can create a wavewform with equal amplitude
		magn = np.mean(np.abs(np.concatenate([fft[:self.Ncarriers//2], fft[self.Ncarriers//2+1:]])))
		print("mean magn", magn)

		# extract DC carrier properties
		c0 = fft[self.Ncarriers // 2]
		print("c0", np.abs(c0) / magn, np.angle(c0) * 360 / (2 * np.pi))

		# show original
		plt.scatter(np.real(fft), np.imag(fft))

		# quantisize phase by fitting it into QPSK
		for i in range(self.Ncarriers):
			# (but leave DC alone)
			if i == self.Ncarriers // 2:
				continue
			phase = np.angle(fft[i])
			# assume QPSK
			phase *= 4 / (2 * np.pi)
			# quantisize phase
			new_phase = ((int(phase) + .5) / 4) * 2 * np.pi
			fft[i] = np.exp(new_phase * 1j) * magn

		# show optimized
		plt.scatter(np.real(fft), np.imag(fft))
		plt.show()

		# create time domain
		td = self.itfft(fft)
		# prefix with CP
		td = np.concatenate((td[-self.cp_length[s]:], td))
		
#		plt.plot(sy)
#		plt.plot(td)
		cor = np.correlate(self.y, td)
		offset = np.argmax(cor)
		print("symbol start at", offset)
		print("we think it starts at", self.sym_offset(s)[0], "so delta", self.sym_offset(s)[0] - offset)
		plt.plot(cor)
		plt.plot(self.y)
		plt.plot(np.linspace(offset, offset + len(td), len(td)), td)
		plt.show()

	def linrot_optimize(self, s):

		sy = self.sym(s, True)
		fft = self.tfft(sy)

		c0_angle = np.exp(1j * 20 / 360 * 2 * np.pi)
		fft /= fft[self.Ncarriers//2] / c0_angle

		res = []

		attempts  = np.linspace(-0.005, 0.005, 1000)
		#attempts = [-0.003033557046979867]
		for linrot in attempts:
			# de-rotate
			fftn = fft * np.exp(2j * np.pi * np.linspace(-.5 * linrot * len(fft), .5 * linrot * len(fft), len(fft)))

			# find QPSK noisyness
			noise = np.sum(((np.angle(fftn) / (2 * np.pi) * 4) % 1 - .5) ** 2)
			res.append(noise)

			#plt.scatter(np.real(fftn), np.imag(fftn))
			#plt.show()

		plt.plot(attempts, res)
#		plt.show()
		
		linrot = attempts[np.argmin(res)]
		fftn = fft * np.exp(2j * np.pi * np.linspace(-.5 * linrot * len(fft), .5 * linrot * len(fft), len(fft)))
#		plt.scatter(np.real(fftn), np.imag(fftn))
#		plt.show()

		print("c0", fftn[self.Ncarriers//2])
		print("c0pol", np.abs(fftn[self.Ncarriers//2]), np.angle(fftn[self.Ncarriers//2]) / (2 * np.pi) * 360)

		return linrot

	def self_corr(self):
		length_to_last = len(self.cp_length[:-1]) * 2048 + sum(self.cp_length[:-1])
		print("length_to_last", length_to_last)

		cor = np.correlate(self.y, self.sym(0), mode='full')
		plt.plot(np.abs(cor))
#		offset = 30720
#		plt.plot(self.y)
#		plt.plot(np.linspace(offset, offset + len(self.y), len(self.y)), self.y)
		plt.show()

	def save(self, linrot):
		print("saving data...")

		for s in range(len(self.cp_length)):
			if s in self.zc_sym:
				continue
			sy = self.sym(s, True)
			fft = self.tfft(sy) # / self.magest
			fft *= np.exp(2j * np.pi * np.linspace(-.5 * linrot * len(fft), .5 * linrot * len(fft), len(fft)))
			c0_angle = np.exp(1j * 20 / 360 * 2 * np.pi)
			fft /= fft[self.Ncarriers//2] / c0_angle

			with open("out/pkt_%d_sym_%d.txt" % (self.pktnum, s), "w") as fo:
				for i in fft:
					fo.write("%f %f\n" % (np.real(i), np.imag(i)))

if True:
	# regular downlink 20MHz
	capture = SpectrumCapture("with_rc_2_2450000.0kHz_200.0MHz.iq") # Fcarrier=22.5MHz
	BW = 20
	Fcarrier = 22.46e6
	Fcarrier -= 5600 - 20 # hand-tuned
	Fcarrier += 45000

	# .12 is fined tuned to find downlink but skip uplink
	# this capture starts in the middle of a packet so ignore first 400k samples
	npkts = 100
	capture.coarse_collect_packets(.12, npkts, start = 400e3 / capture.Fs)

	pkts = range(npkts)

	# pkt0:
	# 471:742
	# 471:0
	# ???

	# pkt 1:
	# 471:538
	# 1:1197
	# 471:538
	pktlength = 1.2e-3
else:
	# DroneID stuff
	capture = SpectrumCapture("pkt_new_1", dtype="<f")

	BW = 10
	Fcarrier = 11.675e6
	Fcarrier += 4.7e6 + 120e3
	Fcarrier += 3213.23

	capture.coarse_collect_packets(.002, 10, start = 0.3e-3)
	pktlength = .8e-3
	pkts = [0]

#capture.plot_spectrum()

for p in pkts:
	print("pkt", p)
	pkt = capture.pkt(p, Fcarrier=Fcarrier, BW=BW, length = pktlength)

#	pkt.plot()
	# correct FFO and restart
	print("FFO start", pkt.detected_ffo)
	Fcarrier += pkt.detected_ffo
	pkt = capture.pkt(p, Fcarrier=Fcarrier, BW=BW, length = pktlength)
	Fcarrier += pkt.detected_ffo
	pkt = capture.pkt(p, Fcarrier=Fcarrier, BW=BW, length = pktlength)
	print("FFO left", pkt.detected_ffo)

#	pkt.self_corr()

#	for p in range(3):
	pkt.find_zc(pkt.zc_sym[0])
#		print("ZC%d %d:%d" % (p, pkt.zc_seq, pkt.zc_cyc))
	pkt.time_domain_zc_corr_x(pkt.zc_sym[0], pkt.zc_seq, pkt.zc_cyc)

#	pkt.linrot_time_domain_corr(1)
	linrot = pkt.linrot_optimize(1)

	pkt.save(linrot)

#	pkt.interactive()

#	exit()

plt.show()
