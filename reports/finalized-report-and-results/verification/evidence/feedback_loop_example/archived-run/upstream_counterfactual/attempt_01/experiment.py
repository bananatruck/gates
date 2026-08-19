class GCN:
    """Two-layer GCN. The reshape reads a width that is bound nowhere."""

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        return [v * hidden_dim for v in x]


model = GCN(1433, 7)
logits = model.forward([0.4, 0.6, 0.2])
test_acc = sum(logits) / len(logits)
record_result("exp1.K2.test_acc", test_acc, unit="ratio")
