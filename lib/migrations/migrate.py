from lib.migrations.migration_v20 import migrate_to_2_0, migrate_to_3_0

def migrate():
    migrate_to_2_0()
    migrate_to_3_0()