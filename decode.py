# decoding with the model

class Decoder(object):
    def __init__(self, opts, models):
        self.opts = opts
        if type(models) not in [list, tuple]:
            models = [models]
        self.mms = models

class SamplingDecoder(Decoder):
    def __init__(self, opts, models):
        super(SamplingDecoder, self).__init__(opts, models)

class BeamDecoder(Decoder):
    def __init__(self, opts, models):
        super(BeamDecoder, self).__init__(opts, models)
