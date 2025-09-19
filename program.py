""" 
    This program combines the jupyter notebook example_demregpy_aiapxl 
    from the demreg git repo into a single python program
""" 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import glob
import os
cwd = os.getcwd()
import sys 
sys.path.append(cwd + '/python')
from dn2dem_pos import dn2dem_pos
import astropy.time as atime
from astropy import units as u
import sunpy.map 
from aiapy.calibrate.util import get_pointing_table, get_correction_table
from aiapy.calibrate import register, update_pointing, degradation
from sunpy.net import Fido, attrs as a
import warnings
warnings.simplefilter('ignore')
plt.rcParams['font.size'] = 16
data_dir = cwd + '/raw_data/'

# --- Load and prepare AIA data ---
wvsrch = a.Wavelength(94 * u.angstrom, 335 * u.angstrom)
ff = sorted(glob.glob(data_dir + 'aia*lev1*.fits'))
if not ff:
    result = Fido.search(a.Time('2010-11-03T12:15:09', '2010-11-03T12:15:19'), a.Instrument("aia"), wvsrch)
    res = Fido.fetch(result, path=data_dir, max_conn=len(result))
    ff = sorted(glob.glob(data_dir + 'aia*lev1*.fits'))
amaps = sunpy.map.Map(ff)
wvn0 = [m.meta['wavelnth'] for m in amaps]
srt_id = sorted(range(len(wvn0)), key=wvn0.__getitem__)
amaps = [amaps[i] for i in srt_id]
ffin = sorted(glob.glob(data_dir + 'aia_smd_*.fits'))
if not ffin:
    start_time, end_time = amaps[0].date, amaps[-1].date
    expanded_start = start_time - 2 * u.hour
    expanded_end = end_time + 2 * u.hour
    pointing_table = get_pointing_table("jsoc", time_range=(expanded_start, expanded_end))
    aprep = [register(update_pointing(m, pointing_table=pointing_table)) for m in amaps]
    seq = sunpy.map.Map(*aprep, sequence=True, sortby=None)
    seq.save(data_dir + 'aia_smd_{index:03}.fits', overwrite='True')
    ffin = sorted(glob.glob(data_dir + 'aia_smd_*.fits'))
aprep = sunpy.map.Map(ffin)
durs = np.array([m.meta['exptime'] for m in aprep])
wvn = np.array([m.meta['wavelnth'] for m in aprep])


# --- Load temperature response ---
trin = io.readsav(cwd + '/aia_tresp_en.dat')
tresp_logt = np.array(trin['logt'])
nt = len(tresp_logt)
nf = len(trin['tr'][:])
trmatrix = np.stack([trin['tr'][i] for i in range(nf)], axis=1)


# --- Degradation correction factors ---
channels = [94, 131, 171, 193, 211, 335] * u.angstrom
time = atime.Time('2010-11-03T12:15:00', scale='utc')
correction_table = get_correction_table(cwd + '/aia_V10_20201119_190000_response_table.txt')
degs = np.array([degradation(ch.value * u.angstrom, time, correction_table=correction_table) for ch in channels])


# --- Uncertainty and error ---
gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6])
dn2ph = gains * np.array([94, 131, 171, 193, 211, 335]) / 3397.
rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])
num_pix = 1

# --- Temperature bins ---
temps = np.logspace(5.7, 7.6, num=42)

# --- DEM calculation for the whole image ---
ny, nx = aprep[0].data.shape
nf = len(aprep)

# Ensure all maps have the same shape before stacking
base_shape = aprep[0].data.shape

if len(sys.argv) == 2:
    data_list = [m.data[:int(sys.argv[1]), :int(sys.argv[1])] for m in aprep]
elif (len(sys.argv) == 3):
    data_list = [m.data[int(sys.argv[1]):int(sys.argv[2]), int(sys.argv[1]):int(sys.argv[2])] for m in aprep]
else: 
    data_list = [m.data if m.data.shape == base_shape else np.resize(m.data, base_shape) for m in aprep]
data_cube = np.stack(data_list, axis=-1)
degs_arr = degs.reshape((1, 1, nf))
durs_arr = durs.reshape((1, 1, nf))
dn2ph_arr = dn2ph.reshape((1, 1, nf))
rdnse_arr = rdnse.reshape((1, 1, nf))
cor_data = data_cube / degs_arr
dn_in = cor_data / durs_arr
shotnoise = (dn2ph_arr * data_cube * num_pix) ** 0.5 / dn2ph_arr / num_pix / degs_arr
edn_in = (rdnse_arr ** 2 + shotnoise ** 2) ** 0.5 / durs_arr
if len(sys.argv) > 2:
    dem_cube, _, _, _, _ = dn2dem_pos(
        dn_in, edn_in, trmatrix, tresp_logt, temps, dem_norm0=np.ones(1), max_iter=10000, target=sys.argv[2]
    )
else:
    dem_cube, _, _, _, _ = dn2dem_pos(
        dn_in, edn_in, trmatrix, tresp_logt, temps, dem_norm0=np.ones(1), max_iter=10000
    )
peak_bin = np.argmax(trmatrix.sum(axis=0))
dem_result_map = dem_cube[:, :, peak_bin]


# --- Plot DEM map ---
dem_map = sunpy.map.Map(dem_result_map, aprep[0].meta)
plt.figure(figsize=(8, 8))
ax = plt.subplot(projection=dem_map)
im = dem_map.plot()
plt.title('DEM Result Map (Peak Bin)')
plt.colorbar(im)
plt.show(block=True)
