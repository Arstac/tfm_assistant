from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
import warnings

class BERTQA:
    def __init__(self, use_finetuned=True):
        self.model = None
        self.tokenizer = None
        
        try:
            if use_finetuned:
                self._load_finetuned_model()
            else:
                self._load_base_model()
                
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer
            )
        except Exception as e:
            warnings.warn(f"Error cargando modelo: {str(e)}")
            self._load_base_model()

    def _load_finetuned_model(self):
        self.model = BertForQuestionAnswering.from_pretrained("models/finetuned_bert")
        self.tokenizer = BertTokenizer.from_pretrained("models/finetuned_bert")

    def _load_base_model(self):
        base_model_name = "mrm8488/bert-spanish-cased-finetuned-qa"
        self.model = BertForQuestionAnswering.from_pretrained(base_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name)

    def answer(self, context: str, question: str) -> dict:
        try:
            return self.qa_pipeline(question=question, context=context)
        except Exception as e:
            return {"error": str(e)}