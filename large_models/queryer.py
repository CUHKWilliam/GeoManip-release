from utils.registry import GENERATORS
from openai import OpenAI
import time

@GENERATORS.register_module()
class WebQueryer:
    def __init__(self, config):
        self.config = config
        self.model = config['model']
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.client = OpenAI(api_key=config['api_key'])

    def query(self, messages, stream=False):
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