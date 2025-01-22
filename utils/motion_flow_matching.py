import torch as th
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

class MotionFlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
        self.path = CondOTProbPath()
        self.model = None
        self.solver = None
    def set_model(self, model):
        self.model = model
        # self.solver = ODESolver(velocity_model=lambda x, t, cond: self.model(x, t, y=cond)[0])
        
        def velocity_fn(t, x, **kwargs):
            t_batch = t.repeat(x.shape[0]) if t.dim() == 0 else t
            cond = {
                'y': {
                    'frame_labels': kwargs['kwargs']['y']['frame_labels'],
                    'mask': None
                }
            }
            return self.model(x, t_batch, y=cond['y'])[0]
            
        self.solver = ODESolver(velocity_model=velocity_fn)

    def forward(self, x1, cond):
        """Calculate training losses for a batch."""
        t = th.rand(x1.shape[0], device=x1.device)
        
        # Flow matching path
        x0 = th.randn_like(x1)
        path_batch = self.path.sample(x0, x1, t)

        # Model prediction
        v_pred, model_cond = self.model(path_batch.x_t, t, y=cond['y'])
        
        # Losses
        fm_loss = F.mse_loss(v_pred, path_batch.dx_t)
        cls_loss = F.cross_entropy(
            model_cond.permute(0,2,1).reshape(-1, model_cond.size(1)),
            cond['y']['frame_labels'].permute(0,2,1).reshape(-1,3)
        )
        
        return {
            'fm_loss': fm_loss,
            'cls_loss': cls_loss,
            'loss': fm_loss + cls_loss
        }

    def sample(self, batch_size, motion_shape, cond, num_steps=10):
        """Sample from the flow matching model."""
        # solver = ODESolver(velocity_model=self.model)
        device = next(self.model.parameters()).device
        x_init = th.randn(*motion_shape, device=device)

        time_grid = th.linspace(0, 1, steps=num_steps, device=x_init.device)
        
        # model_cond = {'y': {'mask': None,'frame_labels': cond}}

        generated = self.solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method='dopri5',
            step_size=None,    # None -> let dopri5 adapt step size internally.
            atol=1e-4,
            rtol=1e-4,
            kwargs=cond
        )
        
        return generated