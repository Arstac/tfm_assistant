from typing import Dict

class Validator:
    @staticmethod
    def cross_validate(regex_data: Dict, bert_data: Dict) -> Dict:
        """Compara resultados y genera alertas."""
        discrepancies = {}
        for key in regex_data:
            if key in bert_data:
                regex_val = str(regex_data[key]).lower().strip(" €")
                bert_val = str(bert_data[key]).lower().strip(" €")
                
                if regex_val != bert_val:
                    discrepancies[key] = {
                        "regex": regex_data[key],
                        "bert": bert_data[key],
                        "confidence": "⚠️ Revisión manual requerida"
                    }
        return discrepancies

    @staticmethod
    def auto_correct(discrepancies: Dict) -> Dict:
        """Correcciones automáticas basadas en reglas."""
        corrections = {}
        for key, vals in discrepancies.items():
            # Ejemplo: Preferir regex para valores numéricos
            if "mes" in vals["regex"] or "eur" in vals["regex"]:
                corrections[key] = vals["regex"]
            else:
                corrections[key] = f'{vals["bert"]} (revisar)'
        return corrections