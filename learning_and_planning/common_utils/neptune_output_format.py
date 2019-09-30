from baselines.logger import KVWriter


class Neptune2OutputFormat(KVWriter):

    def __init__(self, neptune_ctx):
        self.neptune_ctx = neptune_ctx

    def writekvs(self, kvs: dict):
        x = kvs['update_no']
        for k, v in kvs.items():
            self.neptune_ctx.channel_send(k, x=x, y=v)

    def close(self):
        # TODO
        pass
