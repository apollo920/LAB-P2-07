import json
import os
import random

def generate_instruction_pairs(count=50):

    topics = [
        "Design Patterns", "Clean Code", "SOLID Principles", 
        "Unit Testing", "CI/CD Pipelines", "Microservices", 
        "Agile Methodologies", "Code Review", "Git Workflow"
    ]
    
    verbs = ["Explique", "Descreva", "Como implementar", "Quais as vantagens de", "Dê um exemplo de"]
    
    dataset = []
    for i in range(count):
        topic = random.choice(topics)
        verb = random.choice(verbs)
        
        instruction = f"{verb} o conceito de {topic} no desenvolvimento de software moderno."
        response = f"O conceito de {topic} é fundamental na Engenharia de Software. Ele permite que desenvolvedores criem sistemas mais robustos, escaláveis e fáceis de manter. Por exemplo, ao aplicar {topic}, reduzimos o acoplamento e aumentamos a coesão do código, facilitando testes e evolução contínua."
        
        dataset.append({
            "instruction": instruction,
            "response": response
        })
    return dataset

def save_as_jsonl(data, filename="dataset.jsonl", split_ratio=0.9):

    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Dataset gerado com sucesso: {len(data)} entradas.")
    print(f"Sugestão de divisão: {len(train_data)} para treino e {len(test_data)} para teste.")

if __name__ == "__main__":
    
    instructions = generate_instruction_pairs(50)
    save_as_jsonl(instructions)
