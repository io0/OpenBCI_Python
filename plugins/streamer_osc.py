
# requires python-osc
from pythonosc import osc_message_builder
from pythonosc import udp_client
import plugin_interface as plugintypes
import numpy as np
from scipy.signal import welch
from scipy.stats import zscore, norm
from sklearn.base import BaseEstimator, TransformerMixin
import time
from math import log
from scipy import signal

# Use OSC protocol to broadcast data (UDP layer), using "/openbci" stream. (NB. does not check numbers of channel as TCP server)

class RingBuffer(np.ndarray):
    """A multidimensional ring buffer."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def append(self, x):
        """Adds element x to the ring buffer."""
        x = np.asarray(x)
        self[:, :-1] = self[:, 1:]
        self[:, -1] = x
class Filterer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 class_label=0,
                 epoch=3,
                 filter_order=5,
                 filter_width=1.,
                 nb_chan=2,
                 pred_freq=25,
                 sample_rate=250,
                 ssvep_freq=6,
                 ssvep_range_high=60,
                 ssvep_range_low=6):
        self.count_ = 0
        self.epoch_in_samples = int(sample_rate * epoch)
        self.nb_chan = nb_chan
        self.class_label = class_label
        self.sample_rate = sample_rate
        self.filter_width = filter_width
        self.filter_high = ssvep_freq + filter_width
        self.filter_low = ssvep_freq - filter_width
        self.pred_freq = pred_freq
        self.ssvep_freq = ssvep_freq
        self.ssvep_range_high = ssvep_range_high
        self.ssvep_range_low = ssvep_range_low
    def pred_time(self):
        """
        Increments local counter and checks against pred_freq. If self._count is hit, then reset counter and return
            true, else return false.
        :return:
        """
        self.count_ += 1
        if self.count_ >= self.pred_freq:
            self.count_ = 0
            return True
        return False

    def predict_proba(self, X):
        """
        Return a probability between 0 and 1
        :param X: (array like)
        :return:
        """
        # First we take a welch to decompose the new epoch into fequency and power domains
        freq, psd = welch(X, int(self.sample_rate), nperseg=1024)

        # Then normalize the power.
        # Power follows chi-square distribution, that can be pseudo-normalized by a log (because chi square
        #   is aproximately a log-normal distribution)
        psd = np.log(psd)
        psd = np.mean(psd, axis=0)

        # Next we get the index of the bin we are interested in
        low_index = np.where(freq > self.filter_low)[0][0]
        high_index = np.where(freq < self.filter_high)[0][-1]

        # Then we find the standard deviation of the psd over all bins between range low and high
        low_ssvep_index = np.where(freq >= self.ssvep_range_low)[0][0]
        high_ssvep_index = np.where(freq <= self.ssvep_range_high)[0][-1]

        zscores = np.zeros(psd.shape)
        zscores[low_ssvep_index:high_ssvep_index] = zscore(psd[low_ssvep_index:high_ssvep_index])

        pred = norm.cdf(zscores[low_index:high_index+1].mean())

        if np.isnan(pred):
            return 0.0
        else:
            return pred

class StreamerOSC(plugintypes.IPluginExtended):
    """

    Relay OpenBCI values to OSC clients

    Args:
      port: Port of the server
      ip: IP address of the server
      address: name of the stream
    """
        
    def __init__(self, ip='localhost', port=12345, address="/openbci"):
        # connection infos
        self.ip = ip
        self.port = port
        self.address = address
        self.filters = []
        self.buffer = RingBuffer(np.zeros((2, 2500)))
        self.pred_buffer = RingBuffer(np.zeros((2,3)))
        self.num_samples = 0
        self.num_windows = 0
        self.alpha = 0
        self.beta = 0
    # From IPlugin
    def activate(self):
        if len(self.args) > 0:
            self.ip = self.args[0]
        if len(self.args) > 1:
            self.port = int(self.args[1])
        if len(self.args) > 2:
            self.address = self.args[2]
        # init network
        print("Selecting OSC streaming. IP: " + self.ip + ", port: " + str(self.port) + ", address: " + self.address)
        self.client = udp_client.SimpleUDPClient(self.ip, self.port)
        
        # create filters
        self.filters.append(Filterer(pred_freq=200,
                                     ssvep_freq=7,
                                     epoch=5,
                                     filter_width=0.5))
        
        self.filters.append(Filterer(pred_freq=200,
                                     ssvep_freq=12,
                                     epoch=5,
                                     filter_width=0.5))

    # From IPlugin: close connections, send message to client
    def deactivate(self):
        self.client.send_message("/quit")
    def _filter(self, ch):
        fs_Hz = 250.0
        hp_cutoff_Hz = 1.0
        #print("Highpass filtering at: " + str(hp_cutoff_Hz) + " Hz")
        b, a = signal.butter(2, hp_cutoff_Hz/(fs_Hz / 2.0), 'highpass')
        ch = signal.lfilter(b, a, ch, 0)
        notch_freq_Hz = np.array([60.0])  # main + harmonic frequencies
        for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
            bp_stop_Hz = freq_Hz + 3.0*np.array([-1, 1])  # set the stop band
            b, a = signal.butter(3, bp_stop_Hz/(fs_Hz / 2.0), 'bandstop')
            ch = signal.lfilter(b, a, ch, 0)
            #print("Notch filter removing: " + str(bp_stop_Hz[0]) + "-" + str(bp_stop_Hz[1]) + " Hz")
        return ch
        
    # send channels values
    def __call__(self, sample):
        # silently pass if connection drops
        try:
            self.buffer.append(sample.channel_data[:2])
            self.num_samples +=1
            if self.num_samples > 1250 and self.num_samples < 5000:
                if (self.num_samples % 250 == 0):
                    # First we take a welch to decompose the new epoch into fequency and power domains
                    ch = self._filter(self.buffer[:,-1250:])
                    freq, psd = welch(ch, int(self.sample_rate), nperseg=1024)
                    
                    # Then normalize the power.
                    # Power follows chi-square distribution, that can be pseudo-normalized by a log (because chi square
                    #   is aproximately a log-normal distribution)
                    #psd = np.log(psd)
                    low_index = np.where(freq > 16)[0][0]
                    high_index = np.where(freq < 24)[0][-1]
                    beta = np.mean(psd[low_index:high_index])
                    low_index = np.where(freq > 7)[0][0]
                    high_index = np.where(freq < 13)[0][-1]
                    
                    alleft = np.mean(psd[0,low_index:high_index])
                    alright = np.mean(psd[1, low_index:high_index])
                    if alright == 0:
                        ratio = -1
                    else:
                        ratio = log(alleft/alright)
                    print("left: %f, right: %f, asym:%f" %(alleft, alright, ratio))
                    '''
                    psd = np.mean(psd, axis=0)
                    
                    # Next we get the index of the bin we are interested in
                    low_index = np.where(freq > 16)[0][0]
                    high_index = np.where(freq < 24)[0][-1]
                    beta = np.mean(psd[low_index:high_index])
                    low_index = np.where(freq > 7)[0][0]
                    high_index = np.where(freq < 13)[0][-1]
                    alpha = np.mean(psd[low_index:high_index])
                    print("alpha: %f, beta: %f" % (alpha, beta))
                    self.alpha += alpha
                    self.beta += beta
                    self.num_windows +=1
                    '''
            elif self.num_samples == 5000:
                print("alpha av: %f, beta av: %f"% (self.alpha/self.num_windows, self.beta/self.num_windows))
                self.num_windows = 0
                self.alpha = 0
                self.beta = 0
            if self.num_samples == 5500:
                self.num_samples = 0
            #self.client.send_message(self.address, sample.channel_data)
        except:
            return
        

    def show_help(self):
        print("""Optional arguments: [ip [port [address]]]
            \t ip: target IP address (default: 'localhost')
            \t port: target port (default: 12345)
            \t address: select target address (default: '/openbci')""")
