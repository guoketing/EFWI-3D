

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(abs_path, 'Src_3d')
os.makedirs(path+'/build/', exist_ok=True)

setup(
        name='fwi_ops',
        ext_modules=[
            CUDAExtension(
                    name='fwi',
                    sources=[path+'/Torch_Fwi.cpp', path+'/Parameter.cpp', path+'/libCUFD.cu', path+'/stress.cu',
                                    path+'/velocity.cu', path+'/stress_adj.cu', path+'/velocity_adj.cu',
                                    path+'/Model.cu', path+'/Cpml.cu', path+'/utilities.cu', path+'/Src_Rec.cu',
                                    path+'/Boundary.cu'],
                    extra_compile_args={'cxx': ['-g'],
                                        'nvcc': ['-O2']})
        ],
        cmdclass={
            'build_ext': BuildExtension
        })

# def load_fwi(path):
#     fwi = load(name="fwi", sources=[path+'/Torch_Fwi.cpp', path+'/Parameter.cpp', path+'/libCUFD.cu', path+'/stress.cu',
#                                     path+'/velocity.cu', path+'/stress_adj.cu', path+'/velocity_adj.cu',
#                                     path+'/Model.cu', path+'/Cpml.cu', path+'/utilities.cu', path+'/Src_Rec.cu',
#                                     path+'/Boundary.cu'],
#             extra_cflags=[
#                 '-O3 -fopenmp -lpthread'
#             ],
#             extra_include_paths=['/usr/local/cuda/include', path+'/rapidjson'],
#             extra_ldflags=['-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft'],
#             build_directory=path+'/build/',
#             verbose=True)
#     return fwi
