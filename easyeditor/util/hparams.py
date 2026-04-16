import json
from dataclasses import dataclass
from dataclasses import asdict


@dataclass
class HyperParams:




    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)

    def construct_float_from_scientific_notation(config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                try:
                                                                                    
                    config[key] = float(value)
                except:
                    pass
        return config
    
    def to_dict(config) -> dict:
        dict = asdict(config)
        return dict
            
        

                  
                                                       
     
                                                 
                                                                   
                                                
     
                                                                                                                      
     
                                                              
     
                                       
