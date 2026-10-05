from src.db import SessionLocal, AvailableModel, init_db

init_db()  

with SessionLocal() as db:
    model = AvailableModel(
        name="EdgeConv_v1",
        path="models/EdgeConv_best.pt",
        auc_roc=0.9655,
    )
    db.add(model)
    db.commit()
    print(f"ID={model.id}")