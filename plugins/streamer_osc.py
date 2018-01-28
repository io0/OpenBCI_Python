
# requires python-osc
from pythonosc import osc_message_builder
from pythonosc import udp_client
import plugin_interface as plugintypes
import numpy as np
from scipy.signal import welch
from scipy.stats import zscore, norm
from sklearn.base import BaseEstimator, TransformerMixin

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
        self.filters.append(Filterer(pred_freq=100,
                                     ssvep_freq=7,
                                     epoch=5,
                                     filter_width=0.5))
        
        self.filters.append(Filterer(pred_freq=100,
                                     ssvep_freq=12,
                                     epoch=5,
                                     filter_width=0.5))

    # From IPlugin: close connections, send message to client
    def deactivate(self):
        self.client.send_message("/quit")
        
    # send channels values
    def __call__(self, sample):
        # silently pass if connection drops
        try:
            self.buffer.append(sample.channel_data[:2])
            self.num_samples +=1
            pred_ = []
            for filter_ in self.filters:
                if filter_.pred_time():
                    if self.num_samples > filter_.epoch_in_samples:
                        pred = filter_.predict_proba(self.buffer[:, -filter_.epoch_in_samples:])
                        pred_.append(pred)
                        if (pred > 0.87):
                            print(filter_.ssvep_freq)
            self.pred_buffer.append(pred_)
            
            self.client.send_message(self.address, sample.channel_data)
        except:
            return

    def show_help(self):
        print("""Optional arguments: [ip [port [address]]]
            \t ip: target IP address (default: 'localhost')
            \t port: target port (default: 12345)
            \t address: select target address (default: '/openbci')""")
