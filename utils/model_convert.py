

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def convert_models_to_half(model):
    for p in model.parameters():
        p.data = p.data.half()
        if p.grad:
            p.grad.data = p.grad.data.half()