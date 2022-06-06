import microtool
import numpy as np

# Loading the tissuemodel
relaxation_model = microtool.tissue_model.RelaxationTissueModel(t1=900, t2=90)
print(relaxation_model)

tr = np.array([500, 500, 500, 500, 500, 500, 500, 500])
te = np.array([10, 10, 10, 10, 20, 20, 20, 20])
ti = np.array([50, 100, 150, 200, 250, 300, 350, 400])

ir_scheme = microtool.acquisition_scheme.InversionRecoveryAcquisitionScheme(tr, te, ti)
print(ir_scheme)
signal = relaxation_model(ir_scheme)
relaxation_model.fit(ir_scheme, signal)