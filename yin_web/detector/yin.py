
import numpy as np
import scipy.signal as sg


def _next_power_of_2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def difference_vectorized(x, W, max_tau):

    n_fft = _next_power_of_2(2 * W)
    x_pad = np.zeros(n_fft)
    x_pad[:W] = x

    # Autocorrelation via FFT: acf[tau] = sum(x[j]*x[j+tau], j=0..W-1-tau)
    X = np.fft.rfft(x_pad)
    acf = np.fft.irfft(X * np.conj(X), n_fft)[:W]

    # Cumulative sum of x^2 for fast energy lookups
    x_sq_cs = np.zeros(W + 1)
    np.cumsum(x ** 2, out=x_sq_cs[1:])   # x_sq_cs[i] = sum(x[0:i]^2)

    taus = np.arange(max_tau + 1)

    # Energy: sum(x[j]^2, j=1..W-tau-1) = x_sq_cs[W-tau] - x_sq_cs[1]
    e1 = x_sq_cs[W - taus] - x_sq_cs[1]

    # Energy: sum(x[j+tau]^2, j=1..W-tau-1) = sum(x[k]^2, k=1+tau..W-1)
    #        = x_sq_cs[W] - x_sq_cs[1+tau]
    e2 = x_sq_cs[W] - x_sq_cs[1 + taus]

    # Cross-correlation (starting from j=1): acf[tau] - x[0]*x[tau]
    cross = acf[:max_tau + 1] - x[0] * x[:max_tau + 1]

    diff = e1 + e2 - 2.0 * cross
    return diff



def computeCmndf(x, W, min_tau, max_tau):
    diff = difference_vectorized(x, W, max_tau)

    cumsum_all = np.cumsum(diff)

    taus = np.arange(min_tau, max_tau)
    running_sums = cumsum_all[taus]
    cumulative_means = running_sums / taus

    # Avoid division by zero (same as the original if-check)
    cmndf = np.where(cumulative_means == 0, 1.0,
                      diff[taus] / cumulative_means)

    return cmndf



def find_first_local_min_below_threshold(array, threshold):
    # Find all local minima
    local_minima_indices = sg.argrelmin(array)[0]
    for idx in local_minima_indices:
        # Return the FIRST dip that drops below threshold
        if array[idx] < threshold:
            return idx
    return None



def parabolic_interp(y1, y2, y3):
    # Finds the fractional offset for the exact bottom of a parabola
    return 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)


def _search_neighborhood(cmndf, center_idx, radius=2):
    """Return (best_idx, best_val) in ±radius around center_idx."""
    best_idx = center_idx
    best_val = cmndf[center_idx]
    for offset in range(-radius, radius + 1):
        ni = center_idx + offset
        if 0 <= ni < len(cmndf) and cmndf[ni] < best_val:
            best_val = cmndf[ni]
            best_idx = ni
    return best_idx, best_val

def octave_correct(cmndf, predicted_idx, min_tau, threshold):
    candidate_val = cmndf[predicted_idx]

    # Very confident detection — no correction needed
    if candidate_val < 0.01:
        return predicted_idx

    candidate_tau = min_tau + predicted_idx  # actual lag in samples
    corrected_idx = predicted_idx
    corrected_val = candidate_val

    # ---- Downward check (doubled lag = lower octave) ----
    # ONLY if the detection is uncertain — confident fundamentals are trusted.
    if candidate_val > 0.15:
        doubled_idx = min_tau + 2 * predicted_idx
        if doubled_idx < len(cmndf):
            down_idx, down_val = _search_neighborhood(cmndf, doubled_idx)
            if (down_val < candidate_val and down_val < threshold
                    and (candidate_val / max(down_val, 1e-6)) > 5.0):
                corrected_idx = down_idx
                corrected_val = down_val

    # ---- Upward check (halved lag = higher octave) ----
    if corrected_val > 0.02:
        corrected_tau = min_tau + corrected_idx
        halved_tau = corrected_tau // 2
        halved_idx = halved_tau - min_tau
        if 1 <= halved_idx < len(cmndf) - 1:
            up_idx, up_val = _search_neighborhood(cmndf, halved_idx)
            if up_val < corrected_val and up_val < threshold:
                ratio = corrected_val / max(up_val, 1e-6)
                if ratio > 2.0:
                    corrected_idx = up_idx

    return corrected_idx
    


# ----------------------------------------------------------------------------
# CELL 5 (verbatim)
# ----------------------------------------------------------------------------
def pitchDetect(audio, fs, min_f0, max_f0, W, decimation_factor, cmndf_threshold,
                rms_threshold=0.05):
    res = []

    # Downsample for speed
    downsampled_audio = sg.decimate(audio, decimation_factor, zero_phase=True)
    downsampled_fs = fs // decimation_factor

    # Calculate bounds based on DOWNSAMPLED frequency
    min_tau = downsampled_fs // max_f0
    max_tau = downsampled_fs // min_f0

    # 50% Overlapping frames
    step = W // 2
    length = (len(downsampled_audio) // step - 1) * step

    for start in range(0, length, step):
        x = downsampled_audio[start:start+W]
        if len(x) != W:
            break

        # RMS energy gate — check BEFORE appending anything
        frame_energy = np.sqrt(np.mean(x**2))
        if frame_energy < rms_threshold:
            res.append(None)
            continue

        cmndf = computeCmndf(x, W, min_tau, max_tau)
        predicted_idx = find_first_local_min_below_threshold(cmndf, cmndf_threshold)

        if predicted_idx is not None:
            # Octave correction — prefer fundamental over harmonics
            predicted_idx = octave_correct(cmndf, predicted_idx, min_tau, cmndf_threshold)

            # Safety check so interpolation doesn't crash at the edges
            if 0 < predicted_idx < len(cmndf) - 1:
                y1 = cmndf[predicted_idx - 1]
                y2 = cmndf[predicted_idx]
                y3 = cmndf[predicted_idx + 1]

                interp_add = parabolic_interp(y1, y2, y3)

                # Convert array index back to actual lag
                actual_tau = min_tau + predicted_idx + interp_add
                f0 = downsampled_fs / actual_tau
                res.append(f0)
            else:
                res.append(None)
        else:
            res.append(None)  # Unvoiced

    return np.array(res)