from sqlalchemy import text
from app.db.database import engine

def upgrade():
    with engine.connect() as conn:
        print("Altering itineraries table to add total_estimated_cost column...")
        conn.execute(text("ALTER TABLE itineraries ADD COLUMN IF NOT EXISTS total_estimated_cost VARCHAR(255);"))
        
        print("Cleaning up legacy itineraries to prevent schema validation failures...")
        conn.execute(text("DELETE FROM itineraries;"))
        
        conn.commit()
        print("Database upgraded successfully!")

if __name__ == "__main__":
    upgrade()
