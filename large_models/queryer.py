from utils.registry import GENERATORS
from openai import OpenAI
import time
from tqdm import tqdm
import os
from .ollama_manager import OllamaManager

@GENERATORS.register_module()
class WebQueryer:
    def __init__(self, config):
        self.config = config
        self.model = config['model']
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])

    def query(self, messages, task, stream=False):
        if "o1" not in self.model:
            completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=stream
                )
        else:
            completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=stream
                )
        if not stream:
            reply = completion.choices[0].message.content
            return reply
        else:
            output = ""
            start = time.time()
            for chunk in completion:
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
            return output
    
@GENERATORS.register_module()
class LocalQueyer:
    def __init__(self, config):
        self.config = config
    
    def query(self, messages, stream=False):
        '''
        Query local large model here
        '''
        pass

@GENERATORS.register_module()
class LocalQueryerClient():
    def __init__(self, config):
        self.config = config
        self.communication_file = config['communication_file']
        self.lock_file =config['lock_file']
    
    def query_remote(self, messages, stream):
        pass


    def query(self, messages, stream=False):
        self.query_remote(messages, stream)

global_ollama_manager = None


@GENERATORS.register_module()
class RAGQueryer:
    def __init__(self, config):
        self.config = config
        print(f"RAGQueryer build key-value pairs")
        database_dir = self.config['database_dir']
        key_file_name = self.config['key_file_name']
        value_file_name = self.config['value_file_name']
        self.key_value_pairs = {}
        for task in tqdm(os.listdir(database_dir)):
            key_file = os.path.join(database_dir, task, key_file_name)
            value_file = os.path.join(database_dir, task, value_file_name)
            if not os.path.exists(key_file) or not os.path.exists(value_file):
                continue
            with open(key_file, "r") as f:
                key = f.read()
            with open(value_file, "r") as f:
                value = f.read()
            self.key_value_pairs[key] = value
        self.keys = list(self.key_value_pairs.keys())
        self.key_keywords = []
        for key in self.keys: ## lower case
            self.key_keywords.append(key.lower().split(" "))
        print(f"RAGQueryer build key-value pairs done")
        global global_ollama_manager
        if not self.config['share_ollama'] or global_ollama_manager is None:
            ## start ollama services
            ollama_manager = OllamaManager(self.config['local_model'], self.config['local_service_port'], self.config['service_name'])        
            # Install Ollama if needed
            ollama_manager.install_ollama()
            
            # Pull models
            ollama_manager.pull_model()
            
            # Create and start services
            ollama_manager.create_service()
            self.ollama_manager = ollama_manager
            global_ollama_manager = ollama_manager
        else:
            self.ollama_manager = global_ollama_manager

    def query(self, messages, task, stream=False):
        return self._query(messages, task)

    def _query(self, messages, task,  ):
        query_keywords = task.lower().split(" ")
        ## retrieve the key_keyword with the highest overlap
        max_overlap = 0
        max_overlap_key = None
        
        for key_keyword in self.key_keywords:
            overlap = 0
            for keyword in query_keywords:
                if keyword in key_keyword:
                    overlap += 1
            if overlap > max_overlap:
                max_overlap = overlap
                max_overlap_key = key_keyword
        ## get the value
        value = self.key_value_pairs[" ".join(max_overlap_key)]
        if messages[-1]['content'][0] != "text":
            messages[-1]['content'] = messages[-1]['content'][1:]
        messages[-1]['content'] = [{
            "type": "text","text":
f"""Here is a close example similar to our query case written in the key-value format. 
key (which is the task): {max_overlap_key}.
value (which is the answer to the task): {value}.
"""
        }] + messages[-1]['content'] + [{
            "type": "text","text": f"The task: {task}."
        }]
        respones = self.ollama_manager.query_service(messages)
        return respones