# Ttaining arguments to use over here---- it is as a helper function
from transformers import TrainingArguments

# Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir="./",
        logging_steps=10,
        save_strategy="epoch",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy", # Or "f1", etc.
    )
