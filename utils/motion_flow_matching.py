from flow_matching import FlowMatching
import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionFlowMatching(FlowMatching):
    def __init__(self, sigma=0):
        """
        Args:
            sigma: OT regularization strength
        backbone은 나중에 설정
        """
        self.sigma = sigma
        self.backbone = None

    def set_backbone(self, backbone):
        """backbone model 설정"""
        self.backbone = backbone
        super().__init__(backbone=backbone, sigma=self.sigma)

    def get_motion_fields(self, x0, x1, t, *, model):
        """
        기존 구현과 동일
        """
        if self.backbone is None:
            self.set_backbone(model)  # 더 명시적인 방법으로 backbone 설정
            
        t_ = t.view(-1, 1, 1)
        xt = (1 - t_) * x0 + t_ * x1
        vt = model(xt, t)
        
        if self.sigma > 0:
            ut = x1 - x0
            vt = (ut - vt) / self.sigma**2
        
        return vt

    def get_loss(self, x0, x1, t, model_kwargs=None):
        """
        Calculate Flow Matching loss
        Args:
            x0: starting motion
            x1: target motion
            t: time parameter
            model_kwargs: additional arguments for conditioning
        """
        # Get predicted velocity field
        vt = self.get_motion_fields(x0, x1, t, model=self.backbone)
        
        # Ground truth velocity field (constant)
        vt_true = x1 - x0
        
        # Calculate MSE loss
        loss = F.mse_loss(vt, vt_true)
        
        return loss

    def sample(self, x0, num_steps=100, progress=False, model_kwargs=None):
        """
        Generate motion sequence using Euler integration
        Args:
            x0: initial motion
            num_steps: number of integration steps
            progress: whether to show progress bar
            model_kwargs: additional arguments for conditioning
        """
        device = x0.device
        dt = 1.0 / num_steps
        
        # Initialize trajectory
        xt = x0
        
        for i in range(num_steps):
            t = torch.ones(x0.shape[0], device=device) * i * dt
            
            with torch.no_grad():
                # Get velocity field
                vt = self.get_motion_fields(
                    xt, None, t,
                    model=self.backbone
                )
                
                # Euler integration step
                xt = xt + vt * dt
                
                # Optional: Apply motion constraints here
                # xt = apply_motion_constraints(xt)
        
        return xt

    def forward(self, x0, x1, y=None):
        """
        Training forward pass
        Args:
            x0: starting motion
            x1: target motion
            y: conditioning (optional)
        """
        # Random time sampling
        t = torch.rand(x0.shape[0], device=x0.device)
        
        # Calculate loss
        loss = self.get_loss(x0, x1, t, model_kwargs={'y': y})
        
        return loss