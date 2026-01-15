import decoupler as dc


def get_collecTRI_Human(config):
  net = dc.op.collectri(organism='human')
  # edges = dc.op.get_collectri(organism='human', split_complexes=False)
  return net
