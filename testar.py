try:  
    import mindspore as ms  
    print("? MindSpore", ms.__version__, "funcionando!")  
    import numpy as np  
    x = ms.Tensor([1.0, 2.0, 3.0])  
    print("? Tensor:", x.asnumpy())  
except Exception as e:  
    print("? Erro:", e)  
