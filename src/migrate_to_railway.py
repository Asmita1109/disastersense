"""
DisasterSense | Database Migration
Migrates predictions from local PostgreSQL to Railway PostgreSQL.
"""

import psycopg2
from psycopg2.extras import RealDictCursor

LOCAL_DB = {
    "host"    : "localhost",
    "port"    : 5432,
    "dbname"  : "disastersense",
    "user"    : "postgres",
    "password": "disastersense123",
}

RAILWAY_DB = {
    "host"    : "acela.proxy.rlwy.net",
    "port"    : 31259,
    "dbname"  : "railway",
    "user"    : "postgres",
    "password": "njZPDyGkspAbepjwejAiZivCqiieitod",
}


def migrate():
    print("Connecting to local PostgreSQL...")
    local_conn   = psycopg2.connect(**LOCAL_DB)

    print("Connecting to Railway PostgreSQL...")
    railway_conn = psycopg2.connect(**RAILWAY_DB)

    try:
        # Read all predictions from local
        with local_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM predictions ORDER BY timestamp ASC")
            rows = cur.fetchall()
        print(f"Found {len(rows):,} predictions in local database.")

        # Insert into Railway
        with railway_conn.cursor() as cur:
            inserted = 0
            skipped  = 0
            for row in rows:
                try:
                    cur.execute("""
                        INSERT INTO predictions (
                            prediction_id, timestamp, image_prediction, damage_score,
                            text_prediction, informative_score, severity_score,
                            severity_level, inference_time_ms
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        row["prediction_id"],
                        row["timestamp"],
                        row["image_prediction"],
                        row["damage_score"],
                        row["text_prediction"],
                        row["informative_score"],
                        row["severity_score"],
                        row["severity_level"],
                        row["inference_time_ms"],
                    ))
                    inserted += 1
                except Exception as e:
                    skipped += 1
                    print(f"Skipped row: {e}")

        railway_conn.commit()
        print(f"\nMigration complete!")
        print(f"Inserted : {inserted:,}")
        print(f"Skipped  : {skipped:,}")

        # Verify
        with railway_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM predictions")
            count = cur.fetchone()[0]
        print(f"Total in Railway: {count:,}")

    finally:
        local_conn.close()
        railway_conn.close()


if __name__ == "__main__":
    migrate()
