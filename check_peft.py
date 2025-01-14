# check_peft.py

from peft import PrefixTuningConfig
import inspect

def inspect_prefix_tuning_config():
    # Print the signature of the __init__ method
    signature = inspect.signature(PrefixTuningConfig.__init__)
    print("PrefixTuningConfig __init__ signature:")
    print(signature)
    
    # Print docstring of the __init__ method
    doc = PrefixTuningConfig.__init__.__doc__
    print("\nDocstring:")
    print(doc)



# check_peft.py

from peft import IA3Config
import inspect

def inspect_ia3_config():
    # 打印 __init__ 方法的签名
    signature = inspect.signature(IA3Config.__init__)
    print("IA3Config __init__ signature:")
    print(signature)
    
    # 打印 __init__ 方法的文档字符串
    doc = IA3Config.__init__.__doc__
    print("\nDocstring:")
    print(doc)

if __name__ == "__main__":
    inspect_ia3_config()
