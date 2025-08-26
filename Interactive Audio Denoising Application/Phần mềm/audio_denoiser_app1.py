import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.signal import butter, filtfilt, fftconvolve, stft
from scipy.signal.windows import hann
import logging
import os
import tempfile
import base64

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- Hàm chuyển đổi hình ảnh sang Base64 ---
def image_to_base64(image_path):
    """Chuyển đổi tệp hình ảnh thành chuỗi Base64."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # Đối với JPG, mime type là image/jpeg
        return f"data:image/jpeg;base64,{encoded_string}"
    except FileNotFoundError:
        logging.error(f"Lỗi: Không tìm thấy tệp hình ảnh: {image_path}. Vui lòng đảm bảo tệp nằm cùng thư mục với script.")
        return ""
    except Exception as e:
        logging.error(f"Lỗi khi xử lý tệp hình ảnh {image_path}: {e}")
        return ""

# --- Chuyển đổi tất cả hình ảnh cần thiết sang Base64 khi khởi tạo ứng dụng ---
BASE64_QUY_TRINH_KHU_NHIEU = image_to_base64("Quy trình khử nhiễu.jpg")
BASE64_Y_TUONG_1 = image_to_base64("Ý tưởng 1.jpg") # Not used but kept for consistency if needed later
BASE64_Y_TUONG_2 = image_to_base64("Ý tưởng 2.jpg")
BASE64_Y_TUONG_3 = image_to_base64("Ý tưởng 3.jpg") # Not used but kept for consistency if needed later

class Application():
    def __init__(self, file_path=None, method='fft_numpy'):
        self.file_path = file_path
        self.frame_rate = None
        self.sample_width = None
        self.channels = None
        self.method = method
        self.signal_vector = None
        self.spectrum = None
        self.magnitudes = None
        self.frequencies = None
        self.noise_signal_vector = None
        self.noise_spectrum = None
        self.noise_magnitudes = None
        self.noise_frequencies = None
        self.filtered_signal_vector = None # This will hold the result of the last filter applied
        self.filtered_spectrum = None
        self.filtered_magnitudes = None
        self.filtered_frequencies = None

        if file_path is not None:
            logging.info(f"Ứng dụng đã được khởi tạo với tệp: {self.file_path}")

    def _get_signal_statistics(self, data_array, reference_max_for_ratios=None, include_iqr_bounds=True):
        """
        Tính toán các thống kê biên độ cho một mảng dữ liệu (theo độ lớn biên độ).
        Hàm này không còn được sử dụng để hiển thị bảng trong Gradio nhưng được giữ lại
        để đảm bảo tính toàn vẹn của lớp nếu các phần khác cần nó.
        """
        stats_data = []
        if data_array is None or len(data_array) == 0:
            return []

        abs_data = np.abs(data_array)
        current_max_abs = np.max(abs_data)
        ratio_denominator = reference_max_for_ratios if reference_max_for_ratios is not None and reference_max_for_ratios != 0 else current_max_abs
        
        def safe_ratio(value, denom):
            return value / denom if denom != 0 else 0.0

        stats_data.append(['Mean', np.mean(abs_data), safe_ratio(np.mean(abs_data), ratio_denominator)])
        stats_data.append(['Median (Q2)', np.median(abs_data), safe_ratio(np.median(abs_data), ratio_denominator)])
        stats_data.append(['Min', np.min(abs_data), safe_ratio(np.min(abs_data), ratio_denominator)])
        stats_data.append(['Max', current_max_abs, safe_ratio(current_max_abs, ratio_denominator)])
        
        Q1 = np.percentile(abs_data, 25)
        Q3 = np.percentile(abs_data, 75)
        stats_data.append(['Q1', Q1, safe_ratio(Q1, ratio_denominator)])
        stats_data.append(['Q3', Q3, safe_ratio(Q3, ratio_denominator)])
        
        if include_iqr_bounds:
            IQR = Q3 - Q1
            Q1_minus_1_5_IQR = Q1 - 1.5 * IQR
            Q3_plus_1_5_IQR = Q3 + 1.5 * IQR
            stats_data.append(['IQR', IQR, 'N/A'])
            stats_data.append(['Q1 - 1.5*IQR', Q1_minus_1_5_IQR, safe_ratio(Q1_minus_1_5_IQR, ratio_denominator)])
            stats_data.append(['Q3 + 1.5*IQR', Q3_plus_1_5_IQR, safe_ratio(Q3_plus_1_5_IQR, ratio_denominator)])

        return stats_data


    def read_and_analyze_audio(self):
        """
        Đọc một file âm thanh và chuyển đổi thành tín hiệu số (mảng NumPy).
        """
        log_messages_list = []

        if self.file_path is None:
            log_messages_list.append("Vui lòng tải lên một tệp âm thanh.")
            return None, None, None, "\n".join(log_messages_list)

        audio = AudioSegment.from_file(self.file_path)
        
        if audio.channels > 1:
            log_messages_list.append(f"File âm thanh có {audio.channels} kênh. Đang chuyển đổi sang mono (1 kênh).")
            audio = audio.set_channels(1) 
        
        self.frame_rate = audio.frame_rate
        self.sample_width = audio.sample_width
        self.channels = audio.channels 
        self.signal_vector = np.array(audio.get_array_of_samples(), dtype=np.float64) 
        
        if len(self.signal_vector) > 5 * self.frame_rate * 60:
            log_messages_list.append("Cảnh báo: Tệp âm thanh quá lớn. Có thể gặp lỗi bộ nhớ. Vui lòng thử tệp nhỏ hơn hoặc điều chỉnh tham số.")
            logging.warning(f"Tệp âm thanh rất lớn: {len(self.signal_vector)} mẫu. Có thể gây lỗi bộ nhớ.")

        self.spectrum = self._perform_fourier_transform(self.signal_vector)
        self.magnitudes = self._get_magnitudes(self.spectrum, mag_type='positive')
        self.frequencies = self._get_frequencies(self.spectrum, freq_type='positive')
        
        # Reset filtered signal to original when a new file is loaded
        self.filtered_signal_vector = np.copy(self.signal_vector)
        self.filtered_spectrum = np.copy(self.spectrum)
        self.filtered_magnitudes = np.copy(self.magnitudes)
        self.filtered_frequencies = np.copy(self.frequencies)
        
        log_messages_list.append(f"Tín hiệu đã được tạo với kích thước: {self.signal_vector.shape[0]}")
        log_messages_list.append(f"Tốc độ khung hình (frame rate): {self.frame_rate} Hz")
        log_messages_list.append(f"Độ rộng mẫu (sample width): {self.sample_width} bytes")
        log_messages_list.append(f"Số kênh (channels): {self.channels}")
        
        original_audio_path = self._export_signal_to_temp_audio(self.signal_vector, "original_audio.wav")
        original_signal_plot_figure = self._plot_signal(self.signal_vector, self.frame_rate, 'Tín hiệu Âm thanh trong Miền Thời gian (Gốc)')
        original_spectrum_plot_figure = self._plot_spectrum(self.frequencies, self.magnitudes, 'Phổ Tần Số của Tín hiệu Âm thanh (Gốc)')
        
        return original_audio_path, original_signal_plot_figure, original_spectrum_plot_figure, "\n".join(log_messages_list)

    def analyze_noise(self, start_time_sec, end_time_sec):
        if self.signal_vector is None:
            gr.Warning("Vui lòng tải lên và phân tích tín hiệu ban đầu trước.")
            return None, None, None, ""

        start_sample = int(start_time_sec * self.frame_rate)
        end_sample = int(end_time_sec * self.frame_rate)
        start_sample = max(0, start_sample)
        end_sample = min(len(self.signal_vector), end_sample)
        
        self.noise_signal_vector = self.signal_vector[start_sample:end_sample]
        self.noise_spectrum = self._perform_fourier_transform(self.noise_signal_vector)   
        self.noise_magnitudes = self._get_magnitudes(self.noise_spectrum, mag_type='positive')
        self.noise_frequencies = self._get_frequencies(self.noise_spectrum, freq_type='positive')
        
        noise_log_messages = []
        noise_log_messages.append(f"Đoạn nhiễu được chọn từ {start_time_sec:.2f}s đến {end_time_sec:.2f}s.")
        noise_log_messages.append(f"Số mẫu của đoạn nhiễu: {len(self.noise_signal_vector)}")

        noise_audio_path = self._export_signal_to_temp_audio(self.noise_signal_vector, "noise_audio.wav")
        noise_signal_plot_figure = self._plot_signal(self.noise_signal_vector, self.frame_rate, 'Tín hiệu Nhiễu trong Miền Thời gian')
        noise_spectrum_plot_figure = self._plot_spectrum(self.noise_frequencies, self.noise_magnitudes, 'Phổ Tần Số của Tín hiệu Nhiễu')
        
        return noise_audio_path, noise_signal_plot_figure, noise_spectrum_plot_figure, "\n".join(noise_log_messages)

    def restart_filter(self):
        # Đảm bảo có tín hiệu gốc để khởi động lại
        if self.signal_vector is not None:
            self.filtered_signal_vector = np.copy(self.signal_vector)
            self.filtered_spectrum = self._perform_fourier_transform(self.filtered_signal_vector) # Cần tính lại phổ
            self.filtered_magnitudes = self._get_magnitudes(self.filtered_spectrum, mag_type='positive')
            self.filtered_frequencies = self._get_frequencies(self.filtered_spectrum, freq_type='positive')
        else:
            logging.warning("Không có tín hiệu gốc để khởi động lại bộ lọc.")

    def filter_spectral_subtraction_method(self, frame_length, hop_length, spectral_sub_factor):
        self.restart_filter() # Always restart from original signal
        if self.filtered_signal_vector is None:
            gr.Warning("Vui lòng tải lên và phân tích tín hiệu ban đầu trước.")
            return None, None, None

        signal_to_process = np.copy(self.filtered_signal_vector)
        N = len(signal_to_process)
        
        num_frames = int(np.floor((N - frame_length) / hop_length)) + 1
        window = hann(frame_length)
        
        filtered_signal_output_buffer = np.zeros(N + frame_length, dtype=np.float64) 
        
        noise_threshold = 0.0
        if self.noise_spectrum is not None and len(self.noise_spectrum) > 0:
            noise_magnitude_spectrum_estimate = self._get_magnitudes(self.noise_spectrum, mag_type='positive')
            noise_threshold = np.mean(noise_magnitude_spectrum_estimate) + spectral_sub_factor * np.std(noise_magnitude_spectrum_estimate)
        else:
            noise_segment_length = min(self.frame_rate * 2, N // 10)
            if noise_segment_length > 0:
                initial_segment = signal_to_process[:noise_segment_length]
                initial_spectrum = self._perform_fourier_transform(initial_segment)
                initial_magnitudes = self._get_magnitudes(initial_spectrum, mag_type='positive')
                noise_threshold = np.mean(initial_magnitudes) + spectral_sub_factor * np.std(initial_magnitudes)
            else:
                noise_threshold = 0.0

        window_square_sum = np.zeros(N + frame_length, dtype=np.float64)
        
        for i in range(num_frames):
            start_index = i * hop_length
            end_index_frame = start_index + frame_length
            
            current_frame_raw = signal_to_process[start_index : end_index_frame]
            
            if len(current_frame_raw) < frame_length:
                current_frame = np.pad(current_frame_raw, (0, frame_length - len(current_frame_raw)), 'constant')
            else:
                current_frame = current_frame_raw
                
            current_frame_windowed = current_frame * window
            
            transformed_frame = self._perform_fourier_transform(current_frame_windowed)
            
            magnitude = np.abs(transformed_frame)
            phase = np.angle(transformed_frame)
            
            magnitude_filtered = np.maximum(magnitude - noise_threshold, 0)
            
            filtered_spectrum_frame = magnitude_filtered * np.exp(1j * phase)
            
            filtered_time_frame = self._perform_inverse_fourier_transform(filtered_spectrum_frame, frame_length)
            
            filtered_signal_output_buffer[start_index : end_index_frame] += filtered_time_frame * window
            window_square_sum[start_index : end_index_frame] += window * window

        buffer_sliced = filtered_signal_output_buffer[:N]
        window_sum_sliced = window_square_sum[:N]

        non_zero_indices = window_sum_sliced > np.finfo(float).eps 
        
        final_filtered_signal = np.zeros_like(signal_to_process, dtype=np.float64)
        final_filtered_signal[non_zero_indices] = buffer_sliced[non_zero_indices] / window_sum_sliced[non_zero_indices]

        self.filtered_signal_vector = final_filtered_signal
        self.filtered_spectrum = self._perform_fourier_transform(self.filtered_signal_vector)
        self.filtered_magnitudes = self._get_magnitudes(self.filtered_spectrum, mag_type='positive')
        self.filtered_frequencies = self._get_frequencies(self.filtered_spectrum, freq_type='positive')
        
        filtered_audio_path = self._export_signal_to_temp_audio(self.filtered_signal_vector, "filtered_spectral_subtraction.wav")
        filtered_signal_plot_figure = self._plot_filtered_signal('Tín hiệu Âm thanh trong Miền Thời gian (Gốc và Đã Filter - Trừ phổ)')
        filtered_spectrum_plot_figure = self._plot_filtered_spectrum('Phổ Tần Số của Tín hiệu Âm thanh (Gốc và Đã Filter - Trừ phổ)')
        
        return filtered_audio_path, filtered_signal_plot_figure, filtered_spectrum_plot_figure


    def filter_gaussian_smoothing_method(self, kernel_size, kernel_width):
        self.restart_filter() # Always restart from original signal
        if self.filtered_signal_vector is None:
            gr.Warning("Vui lòng tải lên và phân tích tín hiệu ban đầu trước.")
            return None, None, None

        signal_to_process = np.copy(self.filtered_signal_vector) # Start from original signal

        if kernel_size > 0:
            logging.info("Sử dụng làm mượt Gaussian bằng fftconvolve.")
            kernel = np.exp(-np.linspace(-kernel_width, kernel_width, kernel_size)**2)
            kernel = kernel / np.sum(kernel) 
            self.filtered_signal_vector = fftconvolve(signal_to_process, kernel, mode='same')
        else:
            self.filtered_signal_vector = signal_to_process # No smoothing if kernel_size is 0

        self.filtered_spectrum = self._perform_fourier_transform(self.filtered_signal_vector)
        self.filtered_magnitudes = self._get_magnitudes(self.filtered_spectrum, mag_type='positive')
        self.filtered_frequencies = self._get_frequencies(self.filtered_spectrum, freq_type='positive')
        
        filtered_audio_path = self._export_signal_to_temp_audio(self.filtered_signal_vector, "filtered_gaussian_smoothing.wav")
        filtered_signal_plot_figure = self._plot_filtered_signal('Tín hiệu Âm thanh trong Miền Thời gian (Gốc và Đã Filter - Làm mượt Gaussian)')
        filtered_spectrum_plot_figure = self._plot_filtered_spectrum('Phổ Tần Số của Tín hiệu Âm thanh (Gốc và Đã Filter - Làm mượt Gaussian)')
        
        return filtered_audio_path, filtered_signal_plot_figure, filtered_spectrum_plot_figure
                
    def _export_signal_to_temp_audio(self, signal_vector, file_name):
        """
        Chuyển đổi tín hiệu số (mảng NumPy) thành file âm thanh tạm thời và trả về đường dẫn.
        """
        if self.frame_rate is None or self.sample_width is None or self.channels is None:
            logging.error("Thông tin âm thanh (frame_rate, sample_width, channels) chưa được thiết lập.")
            return None

        max_val = np.max(np.abs(signal_vector))
        if max_val > 0:
            scaled_signal = (signal_vector / max_val * (2**15 - 1)).astype(np.int16)
        else:
            scaled_signal = np.zeros_like(signal_vector, dtype=np.int16)

        audio_segment = AudioSegment(
            scaled_signal.tobytes(),
            frame_rate=self.frame_rate,
            sample_width=self.sample_width,
            channels=self.channels
        )
        
        temp_file = os.path.join(tempfile.gettempdir(), file_name)
        audio_segment.export(temp_file, format="wav")
        return temp_file

    def _get_frequencies(self, spectrum, freq_type):
        fs = self.frame_rate        
        N_output = len(spectrum)  
        frequencies = np.zeros(N_output, dtype=np.float64) 
        delta_f = fs / N_output     
        num_positive_bins = (N_output + 1) // 2 
        
        for k in range(num_positive_bins):
            frequencies[k] = k * delta_f
        
        for k in range(num_positive_bins, N_output):
            frequencies[k] = (k - N_output) * delta_f
        
        if freq_type == 'full':
            return frequencies 
        elif freq_type == 'positive':
            return frequencies[:num_positive_bins] 
        
    def _get_magnitudes(self, spectrum, mag_type):
        if mag_type == 'full_raw':
            return np.abs(spectrum) 
        elif mag_type == 'positive':
            N_output = len(spectrum)                        
            positive_points = int(np.ceil(N_output / 2.0))  
            magnitudes = np.abs(spectrum[:positive_points]) 
            
            if N_output % 2 == 0: 
                magnitudes[1:-1] = magnitudes[1:-1] * 2 
            else: 
                magnitudes[1:] = magnitudes[1:] * 2 
            return magnitudes

    def _dft_loops(self, signal_vector): 
        x = signal_vector
        N = len(x)
        X = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            current_Xk = 0.0 + 0.0j
            for n in range(N):
                twiddle_factor = np.exp(-1j * 2 * np.pi * n * k / N)
                current_Xk += x[n] * twiddle_factor 
            X[k] = current_Xk
        return X
    
    def _dft_matrix_mult(self, signal_vector): 
        x = signal_vector
        N = len(x)
        k_indices = np.arange(N).reshape(N, 1)
        n_indices = np.arange(N).reshape(1, N)
        kn = k_indices * n_indices
        F = np.exp(-1j * 2 * np.pi * kn / N)
        X = F @ x
        return X

    def _fft_radix2_dit_recursive(self, signal_vector): 
        x = signal_vector
        original_N = len(x)
        padded_N = 1
        while padded_N < original_N:
            padded_N *= 2
        if padded_N != original_N:
            x = np.pad(x, (0, padded_N - original_N), 'constant')
    
        def recursive_function(signal_vector):
            x = signal_vector
            N = len(x)

            if N <= 1: 
                return x 

            x_even = recursive_function(x[0::2])    
            x_odd = recursive_function(x[1::2])     

            X = np.zeros(N, dtype=np.complex128) 
            
            for k in range(N // 2):
                twiddle = np.exp(-1j * 2 * np.pi * k / N)
                X[k] = x_even[k] + twiddle * x_odd[k]
                X[k + N // 2] = x_even[k] - twiddle * x_odd[k]
            return X
        return recursive_function(x)
    
    def _fft_radix2_dit_iterative(self, signal_vector):
        x = signal_vector
        original_N = len(x)
        padded_N = 1
        while padded_N < original_N:
            padded_N *= 2
        if padded_N != original_N:
            x = np.pad(x, (0, padded_N - original_N), 'constant')        
        N = len(x) 

        X = x.astype(np.complex128) 
        for i in range(1, N - 1):
            j = 0
            m = i
            p = N >> 1 
            while p > 0:
                j += (m & 1) * p
                m >>= 1
                p >>= 1
            if j > i:
                X[i], X[j] = X[j], X[i]

        num_stages = int(np.log2(N))
        for stage in range(1, num_stages + 1):
            L = 2**stage
            L_half = L // 2
            twiddle_base = np.exp(-2j * np.pi / L)
            for k in range(0, N, L):
                twiddle = 1.0 + 0.0j
                for j in range(L_half):
                    t = twiddle * X[k + j + L_half]
                    u = X[k + j]
                    X[k + j] = u + t
                    X[k + j + L_half] = u - t
                    twiddle *= twiddle_base
        return X 
    
    def _perform_fourier_transform(self, signal_vector):
        if self.method == 'fft_numpy':
            spectrum = np.fft.fft(signal_vector)
        elif self.method == 'dft_loops':
            spectrum = self._dft_loops(signal_vector)
        elif self.method == 'dft_matrix_mult':
            spectrum = self._dft_matrix_mult(signal_vector)
        elif self.method == 'fft_radix2_dit_recursive': 
            spectrum = self._fft_radix2_dit_recursive(signal_vector)
        elif self.method == 'fft_radix2_dit_iterative': 
            spectrum = self._fft_radix2_dit_iterative(signal_vector)
        else: # Default to numpy fft if method is not recognized
            spectrum = np.fft.fft(signal_vector)
        return spectrum

    def _idft_loops(self, spectrum):
        X = spectrum
        N = len(X)
        x_reconstructed = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            current_xn = 0.0 + 0.0j
            for k in range(N):
                twiddle_factor = np.exp(1j * 2 * np.pi * n * k / N) 
                current_xn += X[k] * twiddle_factor
            x_reconstructed[n] = current_xn / N 
        return x_reconstructed

    def _idft_matrix_mult(self, spectrum):
        X = spectrum
        N = len(X)
        k_indices = np.arange(N).reshape(N, 1)
        n_indices = np.arange(N).reshape(1, N)
        kn = k_indices * n_indices
        inv_F = np.exp(1j * 2 * np.pi * kn / N) / N 
        x_reconstructed = inv_F @ X
        return x_reconstructed

    def _ifft_radix2_dit_recursive(self, spectrum, original_len):
        X = spectrum
        padded_N = len(X)
        x_conj = self._fft_radix2_dit_recursive(np.conj(X))
        x_reconstructed = np.conj(x_conj) / padded_N
        if len(x_reconstructed) > original_len:
            x_reconstructed = x_reconstructed[:original_len]
        return x_reconstructed
        
    def _ifft_radix2_dit_iterative(self, spectrum, original_len):
        X = spectrum
        padded_N = len(X)
        x_conj = self._fft_radix2_dit_iterative(np.conj(X))
        x_reconstructed = np.conj(x_conj) / padded_N
        if len(x_reconstructed) > original_len:
            x_reconstructed = x_reconstructed[:original_len]
        return x_reconstructed  
    
    def _perform_inverse_fourier_transform(self, spectrum, original_len):
        if self.method == 'fft_numpy':
            signal_vector = np.fft.ifft(spectrum)
        elif self.method == 'dft_loops':
            signal_vector = self._idft_loops(spectrum)
        elif self.method == 'dft_matrix_mult':
            signal_vector = self._idft_matrix_mult(spectrum)
        elif self.method == 'fft_radix2_dit_recursive':
            signal_vector = self._ifft_radix2_dit_recursive(spectrum, original_len)
        elif self.method == 'fft_radix2_dit_iterative':
            signal_vector = self._ifft_radix2_dit_iterative(spectrum, original_len) 
        else: # Default to numpy ifft if method is not recognized
            signal_vector = np.fft.ifft(spectrum)
        signal_vector = np.real(signal_vector).astype(np.float64) 
        return signal_vector

    def _plot_signal(self, signal, rate, title):
        """
        Vẽ biểu đồ tín hiệu trong miền thời gian và trả về đối tượng Figure.
        """
        time_axis = np.arange(len(signal)) / rate
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_axis, signal)
        ax.set_title(title, color='white')
        ax.set_xlabel('Thời gian (giây)', color='white')
        ax.set_ylabel('Biên độ', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('#333333')
        fig.patch.set_facecolor('#333333')
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 0:
            ax.set_ylim(-max_amplitude * 1.1, max_amplitude * 1.1) 
        ax.grid(True, color='#555555')
        plt.tight_layout() 
        return fig
    
    def _plot_spectrum(self, frequencies, magnitudes, title):
        """
        Vẽ biểu đồ phổ tần số và trả về đối tượng Figure.
        """
        fig, ax = plt.subplots(figsize=(14.5, 6))
        ax.plot(frequencies, magnitudes)
        ax.set_title(title, color='white')
        ax.set_xlabel('Tần số (Hz)', color='white')
        ax.set_ylabel('Biên độ', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('#333333')
        fig.patch.set_facecolor('#333333')
        ax.grid(True, color='#555555')
        ax.set_xlim(0, self.frame_rate / 2)
        max_magnitude = np.max(magnitudes)
        if max_magnitude > 0:
            ax.set_ylim(0, max_magnitude * 1.1)
        plt.tight_layout()
        return fig
    
    def _plot_filtered_signal(self, title):
        """
        Vẽ biểu đồ tín hiệu gốc và tín hiệu đã làm sạch trong miền thời gian trên cùng một đồ thị.
        """
        if self.signal_vector is None or self.filtered_signal_vector is None:
            return None
        time_axis = np.arange(len(self.signal_vector)) / self.frame_rate
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_axis, self.signal_vector, label='Tín hiệu Gốc', color='#1f77b4', linewidth=1) 
        ax.plot(time_axis, self.filtered_signal_vector, label='Tín hiệu Đã Filter', color='#ff7f0e', linewidth=1) 
        ax.set_title(title, color='white')
        ax.set_xlabel('Thời gian (giây)', color='white')
        ax.set_ylabel('Biên độ', color='white')
        ax.grid(True, color='#555555')
        ax.legend(loc='upper left', labelcolor='white', framealpha=0.5, facecolor='#333333')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('#333333')
        fig.patch.set_facecolor('#333333')
        plt.tight_layout() 
        max_amplitude_original = np.max(np.abs(self.signal_vector))
        if max_amplitude_original > 0:
            ax.set_ylim(-max_amplitude_original * 1.1, max_amplitude_original * 1.1) 
        return fig
    
    def _plot_filtered_spectrum(self, title):
        """
        Vẽ biểu đồ phổ tần số của tín hiệu gốc và tín hiệu đã làm sạch trên cùng một đồ thị.
        """
        if self.frequencies is None or self.magnitudes is None or self.filtered_frequencies is None or self.filtered_magnitudes is None:
            return None
        fig, ax = plt.subplots(figsize=(14.5, 6)) 
        ax.plot(self.frequencies, self.magnitudes,
                 label='Phổ Gốc', color='#1f77b4', linewidth=1) 
        ax.plot(self.filtered_frequencies, self.filtered_magnitudes,
                 label='Phổ Đã Làm Sạch', color='#ff7f0e', linewidth=1) 
        ax.set_title(title, color='white')
        ax.set_xlabel('Tần số (Hz)', color='white')
        ax.set_ylabel('Biên độ', color='white')
        ax.grid(True, color='#555555')
        ax.set_xlim(0, self.frame_rate / 2) 
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('#333333')
        fig.patch.set_facecolor('#333333')
        max_magnitude_original = np.max(self.magnitudes)
        if max_magnitude_original > 0:
            ax.set_ylim(0, max_magnitude_original * 1.1)
        ax.legend(loc='upper left', labelcolor='white', framealpha=0.5, facecolor='#333333')
        plt.tight_layout()
        return fig

# --- Gradio Interface Functions ---

current_app = None

def upload_audio_and_analyze(audio_file, selected_method):
    global current_app
    if audio_file is None:
        gr.Warning("Vui lòng tải lên một tệp âm thanh.")
        return None, None, None, None, ""

    current_app = Application(file_path=audio_file.name, method=selected_method) 
    
    audio_path, signal_plot_figure, spectrum_plot_figure, log_messages = current_app.read_and_analyze_audio()
    
    return audio_path, signal_plot_figure, spectrum_plot_figure, current_app, log_messages

def analyze_noise_segment(app_state, start_time_sec, end_time_sec):
    if app_state is None or app_state.signal_vector is None:
        gr.Warning("Vui lòng tải lên và phân tích tín hiệu ban đầu trước.")
        return None, None, None, ""
    
    audio_path, signal_plot_figure, spectrum_plot_figure, noise_stats_log = app_state.analyze_noise(start_time_sec, end_time_sec) 
    return audio_path, signal_plot_figure, spectrum_plot_figure, noise_stats_log

def run_spectral_subtraction(app_state, frame_length, hop_length, spectral_sub_factor):
    if app_state is None or app_state.signal_vector is None:
        gr.Warning("Vui lòng tải lên và phân tích tín hiệu ban đầu trước.")
        return None, None, None
    
    audio_path, signal_plot_figure, spectrum_plot_figure = app_state.filter_spectral_subtraction_method( 
        frame_length=frame_length, hop_length=hop_length, spectral_sub_factor=spectral_sub_factor
    )
    return audio_path, signal_plot_figure, spectrum_plot_figure 

def run_gaussian_smoothing(app_state, kernel_size, kernel_width):
    if app_state is None or app_state.signal_vector is None:
        gr.Warning("Vui lòng tải lên và phân tích tín hiệu ban đầu trước.")
        return None, None, None
    
    audio_path, signal_plot_figure, spectrum_plot_figure = app_state.filter_gaussian_smoothing_method( 
        kernel_size=kernel_size, kernel_width=kernel_width
    )
    return audio_path, signal_plot_figure, spectrum_plot_figure 

# --- Gradio UI Layout ---
custom_css = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #1a1a1a;
    color: #f0f0f0;
}
.gradio-container {
    max-width: 1200px;
    margin: 30px auto;
    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    border-radius: 15px;
    overflow: hidden;
    background-color: #2a2a2a;
    padding: 20px;
}
h1 {
    color: #76e5ff;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 20px;
    text-shadow: 1px 1px 5px rgba(0,200,255,0.3);
}
h3 {
    color: #a0e6ff;
    border-bottom: 2px solid #00bfff;
    padding-bottom: 10px;
    margin-top: 20px;
    margin-bottom: 15px;
}
.gr-group {
    background-color: #3a3a3a;
    border: 1px solid #555555;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.gradio-tabs {
    border-radius: 10px;
    overflow: hidden;
}
.gradio-tab-item {
    padding: 20px;
    background-color: #2a2a2a;
    border-radius: 10px;
}
.gradio-tab-item h3 {
    color: #87ceeb;
    border-bottom: 1px solid #4682b4;
}
.gr-button {
    background-color: #00bfff;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 1.1em;
    transition: background-color 0.3s ease, transform 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.gr-button:hover {
    background-color: #009acd;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
.gr-file, .gr-dropdown, .gr-number, .gr-slider {
    margin-bottom: 15px;
}
.gr-label {
    font-weight: bold;
    color: #bbbbbb;
}
.gr-textbox textarea {
    background-color: #444444 !important;
    color: #eeeeee !important;
    border: 1px solid #666666 !important;
    border-radius: 5px !important;
}
img {
    max-width: 80%;
    height: auto;
    border-radius: 8px;
    margin-top: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    display: block;
    margin-left: auto;
    margin-right: auto;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #76e5ff;'>✨ Ứng dụng Khử Nhiễu Âm thanh ✨</h1>")
    gr.Markdown("""
    <p style='text-align: center; font-size: 1.1em; color: #f0f0f0;'>
    Chào mừng bạn đến với ứng dụng khử nhiễu âm thanh! Tại đây, bạn có thể tải lên các tệp âm thanh, phân tích tín hiệu, xác định các đoạn nhiễu, và áp dụng nhiều phương pháp lọc khác nhau để cải thiện chất lượng âm thanh.
    </p>
    """)
    
    app_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("<h3>Tải lên và Cấu hình ban đầu</h3>")
                audio_input = gr.File(label="Tải lên tệp âm thanh (.mp3, .wav, v.v.)")
                method_selector = gr.Dropdown(
                    choices=['fft_numpy', 'dft_loops', 'dft_matrix_mult', 'fft_radix2_dit_recursive', 'fft_radix2_dit_iterative'],
                    value='fft_numpy',
                    label="Chọn phương pháp biến đổi Fourier"
                )
                upload_btn = gr.Button("Tải lên và Phân tích Tín hiệu Gốc")
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("<h3>Tín hiệu và Phổ ban đầu</h3>")
                original_audio_output = gr.Audio(label="Âm thanh Gốc", type="filepath")
                original_signal_plot = gr.Plot(label="Tín hiệu Gốc (Miền Thời gian)")
                original_spectrum_plot = gr.Plot(label="Phổ Tần Số Gốc")
                analysis_log_output = gr.Textbox(label="Thông tin Phân tích Tín hiệu", interactive=False, lines=5)


    upload_btn.click(
        upload_audio_and_analyze,
        inputs=[audio_input, method_selector],
        outputs=[original_audio_output, original_signal_plot, original_spectrum_plot, app_state, analysis_log_output]
    )

    gr.Markdown("<h2 style='text-align: center; color: #76e5ff; margin-top: 40px;'>Các Quy trình Lọc Nhiễu</h2>")
    gr.Markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="{BASE64_QUY_TRINH_KHU_NHIEU}" alt="[Sơ đồ Quy trình Khử nhiễu]">
        <p style="font-style: italic; color: #bbbbbb; font-size: 0.9em;">Sơ đồ tổng quan về quy trình khử nhiễu.</p>
    </div>
    <p style='text-align: center; font-size: 1.0em; color: #f0f0f0;'>
    Dưới đây là các tab tương ứng với các quy trình lọc nhiễu khác nhau mà bạn có thể áp dụng.
    </p>
    """)

    with gr.Tab("Phân tích đoạn nhiễu"):
        with gr.Group(): 
            gr.Markdown("<p style='color: #f0f0f0; font-size: 0.95em;'>Chọn một đoạn thời gian trong tín hiệu gốc mà bạn tin rằng chỉ chứa nhiễu để ứng dụng có thể học và ước lượng đặc tính nhiễu.</p>")
            with gr.Row():
                noise_start_time = gr.Number(label="Thời gian bắt đầu nhiễu (giây)", value=0)
                noise_end_time = gr.Number(label="Thời gian kết thúc nhiễu (giây)", value=5)
            analyze_noise_btn = gr.Button("Phân tích đoạn nhiễu")
            
            noise_audio_output = gr.Audio(label="Âm thanh Nhiễu", type="filepath")
            noise_signal_plot = gr.Plot(label="Tín hiệu Nhiễu (Miền Thời gian)")
            noise_spectrum_plot = gr.Plot(label="Phổ Tần Số Nhiễu")
            noise_analysis_stats_output = gr.Textbox(label="Thông tin Phân tích Đoạn Nhiễu", interactive=False, lines=5)


        analyze_noise_btn.click(
            analyze_noise_segment,
            inputs=[app_state, noise_start_time, noise_end_time],
            outputs=[noise_audio_output, noise_signal_plot, noise_spectrum_plot, noise_analysis_stats_output]
        )

    with gr.Tab("Thực hiện khử nhiễu"):
        with gr.Tabs() as filtered_tabs:
            with gr.TabItem("Trừ phổ (Spectral Subtraction)"):
                with gr.Group(): 
                    gr.Markdown("<h3>Trừ phổ (Spectral Subtraction)</h3>")
                    gr.Markdown("""
                    <p style='color: #f0f0f0; font-size: 0.95em;'>
                    Phương pháp này chia tín hiệu thành các khung nhỏ (Short-Time Fourier Transform - STFT). Nó ước lượng phổ nhiễu (từ đoạn nhiễu bạn cung cấp hoặc ước lượng tự động) và trừ đi khỏi phổ của tín hiệu.
                    </p>
                    """)
                    gr.Markdown(f"""
                    <div style="text-align: center; margin-bottom: 20px;">
                        <img src="{BASE64_Y_TUONG_2}" alt="[Sơ đồ Trừ phổ]">
                        <p style="font-style: italic; color: #bbbbbb; font-size: 0.9em;">Sơ đồ quy trình khử nhiễu bằng trừ phổ.</p>
                    </div>
                    """)
                    frame_length_ss = gr.Slider(minimum=256, maximum=4096, step=256, value=2048, label="Độ dài khung (samples)")
                    hop_length_ss = gr.Slider(minimum=128, maximum=2048, step=128, value=512, label="Độ dài bước nhảy (samples)")
                    spectral_sub_factor_ss = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.1, label="Hệ số trừ phổ (kiểm soát mức độ trừ nhiễu)")
                    
                    run_ss_btn = gr.Button("Chạy Trừ phổ")

                    filtered_audio_output_ss = gr.Audio(label="Âm thanh đã khử nhiễu (Trừ phổ)", type="filepath")
                    filtered_signal_plot_ss = gr.Plot(label="Tín hiệu đã khử nhiễu (Trừ phổ - Miền Thời gian)")
                    filtered_spectrum_plot_ss = gr.Plot(label="Phổ đã khử nhiễu (Trừ phổ)")

                run_ss_btn.click(
                    run_spectral_subtraction,
                    inputs=[app_state, frame_length_ss, hop_length_ss, spectral_sub_factor_ss],
                    outputs=[filtered_audio_output_ss, filtered_signal_plot_ss, filtered_spectrum_plot_ss]
                )

            with gr.TabItem("Làm mượt Gaussian"):
                with gr.Group():
                    gr.Markdown("<h3>Làm mượt Gaussian</h3>")
                    gr.Markdown("""
                    <p style='color: #f0f0f0; font-size: 0.95em;'>
                    Phương pháp này áp dụng bộ lọc Gaussian kernel để làm mượt tín hiệu, giúp giảm thiểu nhiễu ngẫu nhiên và làm mịn các đỉnh tín hiệu.
                    </p>
                    """)
                    gr.Markdown(f"""
                    <div style="text-align: center; margin-bottom: 20px;">
                        <img src="{BASE64_Y_TUONG_2}" alt="[Sơ đồ Làm mượt Gaussian]">
                        <p style="font-style: italic; color: #bbbbbb; font-size: 0.9em;">Sơ đồ quy trình làm mượt bằng Gaussian.</p>
                    </div>
                    """)
                    kernel_size_gs = gr.Slider(minimum=10, maximum=200, step=10, value=50, label="Kích thước Kernel Gaussian")
                    kernel_width_gs = gr.Slider(minimum=0.5, maximum=5.0, step=0.1, value=3.0, label="Độ rộng Kernel Gaussian")
                    # Removed the gr.Radio component for manual/fftconvolve selection

                    run_gs_btn = gr.Button("Chạy Làm mượt Gaussian")

                    filtered_audio_output_gs = gr.Audio(label="Âm thanh đã khử nhiễu (Làm mượt Gaussian)", type="filepath")
                    filtered_signal_plot_gs = gr.Plot(label="Tín hiệu đã khử nhiễu (Làm mượt Gaussian - Miền Thời gian)")
                    filtered_spectrum_plot_gs = gr.Plot(label="Phổ đã khử nhiễu (Làm mượt Gaussian)")

                run_gs_btn.click(
                    run_gaussian_smoothing,
                    inputs=[app_state, kernel_size_gs, kernel_width_gs], # Removed use_manual_convolution_gs from inputs
                    outputs=[filtered_audio_output_gs, filtered_signal_plot_gs, filtered_spectrum_plot_gs]
                )

demo.launch()
