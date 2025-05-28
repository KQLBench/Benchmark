import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder that handles various data types:
    - datetime objects
    - objects with to_dict or as_dict methods
    - objects that can be converted to dictionaries via __dict__
    - any other object that can be represented as a string
    """
    def default(self, obj):
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
            
        # Handle objects with to_dict or as_dict method
        if hasattr(obj, 'to_dict') and callable(obj.to_dict):
            return obj.to_dict()
        if hasattr(obj, 'as_dict') and callable(obj.as_dict):
            return obj.as_dict()
            
        # If obj has __dict__, convert it to a dictionary
        if hasattr(obj, '__dict__'):
            try:
                return obj.__dict__
            except:
                pass
                
        # For Azure SDK objects, try to extract data attributes
        try:
            # If it's iterable, convert to list
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                return list(obj)
                
            # Try to convert to a basic type
            if hasattr(obj, '__str__'):
                return str(obj)
        except Exception as e:
            logger.debug(f"Error serializing {type(obj).__name__}: {str(e)}")
        
        # Default fallback for unserializable objects    
        return f"<Unserializable object of type {type(obj).__name__}>"

def dump_to_json(data, file_obj, **kwargs):
    """
    Dump data to JSON with enhanced type handling
    
    Args:
        data: Data to dump to JSON
        file_obj: File object to write to
        **kwargs: Additional arguments to pass to json.dump
    """
    json.dump(data, file_obj, cls=EnhancedJSONEncoder, **kwargs)

def dumps_to_json(data, **kwargs):
    """
    Convert data to JSON string with enhanced type handling
    
    Args:
        data: Data to convert to JSON
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string
    """
    return json.dumps(data, cls=EnhancedJSONEncoder, **kwargs)
