import torch
from torch.autograd import Function

__all__ = [
    "clip",
    "lower_bound",
    "upper_bound",
    "quantize",
    "signum"
]


# Define the customed functions
class ClipFunction(Function):
    @staticmethod
    def forward(ctx, input, minval, maxval):
        return input.clamp(minval, maxval)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1] is True or ctx.needs_input_grad[2] is True:
            raise ValueError("Function CLIP: minval or maxval should not require gradients!")

        return grad_output, None, None


class LowerBoundFunction(Function):
    @staticmethod
    def forward(ctx, input, lower_bound):
        # Detach the inputs for better processing.
        input = input.detach()
        bound = torch.tensor(lower_bound, device=input.device, dtype=input.dtype)
        ctx.save_for_backward(input, bound)

        return input.clamp(min=bound)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1] is True:
            raise ValueError("Function LowerBound: lower_bound should not require gradients!")
        # Compute the values
        input, bound = ctx.saved_tensors
        pass_through_if = (input >= bound) | (grad_output < 0)
        grad_input = pass_through_if.to(grad_output.device, grad_output.dtype) * grad_output

        return grad_input, None


class UpperBoundFunction(Function):
    @staticmethod
    def forward(ctx, input, upper_bound):
        # Detach the inputs for better processing.
        input = input.detach()
        bound = torch.tensor(upper_bound, device=input.device, dtype=input.dtype)
        ctx.save_for_backward(input, bound)
        return input.clamp(max=upper_bound)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1] is True:
            raise ValueError("Function LowerBound: lower_bound should not require gradients!")
        # Compute the values
        input, bound = ctx.saved_tensors
        pass_through_if = (input <= bound) | (grad_output > 0)
        grad_input = pass_through_if.to(grad_output.device, grad_output.dtype) * grad_output

        return grad_input, None


class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SignumFunction(Function):
    @staticmethod
    def forward(ctx, input):
        clip_zero = input.detach().clone()
        clip_zero[clip_zero == 0] = 1
        output = torch.sign(clip_zero)

        return output

    @staticmethod
    def backward(ctx, grad_input):
        return torch.zeros_like(grad_input)


# Define the API functions
clip = ClipFunction.apply
lower_bound = LowerBoundFunction.apply
upper_bound = UpperBoundFunction.apply
quantize = QuantizeFunction.apply
signum = SignumFunction.apply

if __name__ == "__main__":
    modes = {
        "dtype": torch.float32,
        "device": torch.device("cuda:0"),
    }

    f_test = lambda x: (torch.round(x) - x).detach() + x

    inputs1 = torch.randn(6, 6, **modes, requires_grad=True)
    inputs2 = inputs1.detach().clone().requires_grad_()

    outputs1 = quantize(inputs1)
    outputs2 = f_test(inputs2)

    gradients = torch.randn_like(inputs1)
    outputs1.backward(gradients)
    outputs2.backward(gradients)

    print("Output error: {:>3.1e};".format(torch.mean((outputs1 - outputs2).pow(2))))
    print("Gradients error: {:>3.1e};".format(torch.mean((inputs1.grad - inputs2.grad).pow(2))))
