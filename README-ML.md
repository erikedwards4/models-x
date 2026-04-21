# models-x
Repo for external models in JAX.  
This README is for machine learning (ML) and JAX points and discussion.  

## Modules vs functions
As in Pytorch, which has classes that inherit from torch.nn.Module,  
and functions that are imported from torch.nn.functional as F,  
here we distinguish classes that resemble nn.Modules in subdir 'nn'  
vs. functions that resemble F in subdir 'fn':  
```
./src/models_x/nn/{embedding,linear,layer_norm}.py
./src/models_x/fn/{relu,gelu,softmax,dropout}.py
```
Note that the 'src/models_x' part is the standard src layout as set by uv.  

This is also fundamental to JAX -- only use classes (nn) if there  
are learnable parameters to initialize within an init_params method.  
Only use a class when it owns params and is to be registered as a Pytree.  
Otherwise, use functions (fn), recalling that the JAX philosophy is  
functional, and this is critical to its tracing, JIT and performance.  

## JAX models
Whereas Pytorch uses a hierarchy of torch.nn.Module members, e.g.:  
```
class GPT2(torch.nn.Module):
    def __init__(self, ...) -> None:
    # Set and check self attributes and sub-modules here
    ...

    def forward(self, ...) -> torch.Tensor:
    # Forward method
```

Instead JAX uses a hierarchy of dataclass members (Python Standard Library),  
and further decorates with @registered_dataclass, which is a JAX-specific  
decorater for the JAX tree_util to manage:  
```
@registered_dataclass
@dataclass(frozen=True)
class GPT2():
    # Get config and register self attributes here
    ...

    def __post_init__(self) -> None:
    # Check self attributes here
    # Also set sub-modules here
    ...

    def init_params(self, ...): -> dict[str, Any]:
    # Set actual params dict here
    ...

    def __call__(self, ...) -> jax.Array:
    # Forward method
    ...
```
Note that post_init is called more often than init_params, so only  
put what is necessary therein. It is also stated that structure is  
static (post_init), whereas params remain dynamic (init_params).  
