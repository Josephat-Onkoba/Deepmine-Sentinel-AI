"""
Database maintenance management command for resolving SQLite locks and optimization.
"""

from django.core.management.base import BaseCommand
from django.db import connection
import sqlite3
import os


class Command(BaseCommand):
    help = 'Optimize SQLite database and resolve lock issues'

    def add_arguments(self, parser):
        parser.add_argument(
            '--vacuum',
            action='store_true',
            help='Run VACUUM to optimize database',
        )
        parser.add_argument(
            '--check-locks',
            action='store_true',
            help='Check for database locks',
        )
        parser.add_argument(
            '--optimize',
            action='store_true',
            help='Run optimization commands',
        )

    def handle(self, *args, **options):
        db_path = connection.settings_dict['NAME']
        
        if options['check_locks']:
            self.check_database_locks(db_path)
        
        if options['vacuum']:
            self.vacuum_database(db_path)
        
        if options['optimize']:
            self.optimize_database(db_path)
        
        if not any([options['vacuum'], options['check_locks'], options['optimize']]):
            self.stdout.write(
                self.style.SUCCESS('Running full database optimization...')
            )
            self.check_database_locks(db_path)
            self.optimize_database(db_path)
            self.vacuum_database(db_path)

    def check_database_locks(self, db_path):
        """Check for database locks and active connections"""
        self.stdout.write('Checking database locks...')
        
        try:
            conn = sqlite3.connect(db_path, timeout=1)
            cursor = conn.cursor()
            
            # Check for active transactions
            cursor.execute("BEGIN IMMEDIATE;")
            cursor.execute("ROLLBACK;")
            
            conn.close()
            self.stdout.write(
                self.style.SUCCESS('✓ Database is accessible, no locks detected')
            )
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self.stdout.write(
                    self.style.ERROR(f'✗ Database is locked: {e}')
                )
                self.stdout.write('Try stopping all Django processes and running again.')
            else:
                self.stdout.write(
                    self.style.ERROR(f'✗ Database error: {e}')
                )

    def vacuum_database(self, db_path):
        """Run VACUUM to optimize database"""
        self.stdout.write('Running VACUUM to optimize database...')
        
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("VACUUM;")
            conn.close()
            
            self.stdout.write(
                self.style.SUCCESS('✓ Database vacuumed successfully')
            )
        except sqlite3.Error as e:
            self.stdout.write(
                self.style.ERROR(f'✗ VACUUM failed: {e}')
            )

    def optimize_database(self, db_path):
        """Run optimization pragmas"""
        self.stdout.write('Optimizing database settings...')
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Optimization commands
            optimizations = [
                "PRAGMA optimize;",
                "PRAGMA analysis_limit=1000;",
                "PRAGMA journal_mode=WAL;",
                "PRAGMA synchronous=NORMAL;",
                "PRAGMA cache_size=10000;",
                "PRAGMA temp_store=MEMORY;",
                "PRAGMA mmap_size=268435456;",  # 256MB
            ]
            
            for pragma in optimizations:
                cursor.execute(pragma)
                result = cursor.fetchone()
                if result:
                    self.stdout.write(f'  {pragma} -> {result[0]}')
            
            conn.close()
            self.stdout.write(
                self.style.SUCCESS('✓ Database optimized successfully')
            )
        except sqlite3.Error as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Optimization failed: {e}')
            )

    def get_database_info(self, db_path):
        """Get database statistics"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get database size
            size = os.path.getsize(db_path)
            self.stdout.write(f'Database size: {size:,} bytes ({size/1024/1024:.1f} MB)')
            
            # Get table count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
            table_count = cursor.fetchone()[0]
            self.stdout.write(f'Tables: {table_count}')
            
            # Get pragma info
            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0]
            self.stdout.write(f'Journal mode: {journal_mode}')
            
            cursor.execute("PRAGMA synchronous;")
            sync_mode = cursor.fetchone()[0]
            self.stdout.write(f'Synchronous mode: {sync_mode}')
            
            conn.close()
            
        except sqlite3.Error as e:
            self.stdout.write(
                self.style.ERROR(f'Could not get database info: {e}')
            )
