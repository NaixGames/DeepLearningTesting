import torch
import Functions

B, C = 5, 10
y = torch.ones(B,C)
y_pred = torch.ones(B,C)


dimL = 40
hL = torch.ones(B,dimL)
U = torch.ones(dimL,C)
c = torch.ones(C)
uLm1 = hL @ U + c

# This is the gradient for the last value we pass to the softmax
dL_duLm1 = Functions.softmax(uLm1, 1) - y

# This dimensions should fit
assert dL_duLm1.size() == uLm1.size()

# This are the gradient with respect to the parameters in the last layer
dL_dU = hL.t() @ dL_duLm1
dL_dc = dL_duLm1.sum(dim=0)
dL_dhL = dL_duLm1 @ U.t() 


# This dimensions should fit
assert dL_dU.size() == U.size()
assert dL_dc.size() == c.size()
assert dL_dhL.size() == hL.size()


# Similar as before for hidden layers
dimk = 20
dimkm1 = 30
hk = torch.ones(B,dimk)
Wk = torch.ones(dimk,dimkm1)
bk = torch.ones(dimkm1)
uk = hk @ Wk + bk
dL_dhkm1 = torch.ones(B,dimkm1)

# This should be the formula for the hidden layers
# Note we need the derivate tensors here
dL_duk = dL_dhkm1 * Functions.sigmoid_der(uk) #Important here, the dL_dhkm1 needs to be the gradient of the next layer!
dL_dWk = hk.t() @ dL_duk
dL_dbk = dL_duk.sum(dim=0)
dL_dhk = dL_duk @ Wk.t()

# This dimensions should fit

assert dL_dWk.size() == Wk.size()
assert dL_dbk.size() == bk.size()
assert dL_dhk.size() == hk.size()