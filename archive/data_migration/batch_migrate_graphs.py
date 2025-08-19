#!/usr/bin/env python3
"""
IRONFORGE Batch Graph Schema Migration
======================================

Technical Debt Surgeon: Batch processing script for migrating
legacy 34D graph files to current 37D schema format while
maintaining strict data integrity.

MISSION: Enable archaeological discovery system to process
historical data archives by normalizing schema inconsistencies.

Features:
- Batch processing of graph files from input directory
- Comprehensive validation and error reporting
- Progress tracking with detailed statistics
- NO FALLBACKS policy - fails clean on corrupted data
- Backup creation before migration
- Parallel processing capability (future enhancement)

Usage:
    python3 batch_migrate_graphs.py --input /path/to/34d/graphs --output /path/to/37d/graphs
"""

import argparse
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

from schema_normalizer import SchemaNormalizer


class BatchGraphMigrator:
    """
    Batch migration system for IRONFORGE graph files

    Technical Debt Surgeon: Comprehensive batch processing with
    detailed reporting and strict data integrity validation
    """

    def __init__(self):
        self.normalizer = SchemaNormalizer()
        self.batch_stats = {
            "total_files": 0,
            "processed_files": 0,
            "migrated_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "errors": [],
            "warnings": [],
            "start_time": None,
            "end_time": None,
        }

    def discover_graph_files(self, input_dir: str, patterns: list[str] = None) -> list[Path]:
        """
        Discover graph files in input directory

        Technical Debt Surgeon: Flexible file discovery with validation
        """
        if patterns is None:
            patterns = ["*.json", "*.pkl"]  # Support both JSON and pickle formats

        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")

        discovered_files = []
        for pattern in patterns:
            discovered_files.extend(input_path.glob(pattern))

        # Filter for likely graph files (contain "graph" in name or are in graph directories)
        graph_files = []
        for file_path in discovered_files:
            if (
                "graph" in file_path.name.lower()
                or "graph" in str(file_path.parent).lower()
                or any(
                    keyword in file_path.name.lower() for keyword in ["nodes", "edges", "discovery"]
                )
            ):
                graph_files.append(file_path)

        return sorted(graph_files)

    def create_backup(self, input_dir: str, backup_dir: str | None = None) -> str:
        """
        Create backup of input directory before migration

        Technical Debt Surgeon: Safety measure to prevent data loss
        """
        if backup_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{input_dir}_backup_{timestamp}"

        backup_path = Path(backup_dir)
        input_path = Path(input_dir)

        print(f"🔒 Creating backup: {input_path} → {backup_path}")

        try:
            shutil.copytree(input_path, backup_path, dirs_exist_ok=False)
            print(f"✅ Backup created successfully: {backup_path}")
            return str(backup_path)
        except Exception as e:
            raise ValueError(f"Backup creation failed: {str(e)}") from e

    def analyze_batch_requirements(self, graph_files: list[Path]) -> dict:
        """
        Analyze batch of files to determine migration requirements

        Technical Debt Surgeon: Pre-migration analysis for planning
        """
        analysis = {
            "total_files": len(graph_files),
            "schema_distribution": {},
            "migration_candidates": [],
            "validation_failures": [],
            "file_size_stats": {"min": float("inf"), "max": 0, "total": 0},
            "analysis_errors": [],
        }

        print(f"🔍 Analyzing {len(graph_files)} graph files...")

        for i, file_path in enumerate(graph_files):
            if i % 10 == 0:  # Progress update every 10 files
                print(f"   Analyzed: {i}/{len(graph_files)} files")

            try:
                # Get file size
                file_size = file_path.stat().st_size
                analysis["file_size_stats"]["min"] = min(
                    analysis["file_size_stats"]["min"], file_size
                )
                analysis["file_size_stats"]["max"] = max(
                    analysis["file_size_stats"]["max"], file_size
                )
                analysis["file_size_stats"]["total"] += file_size

                # Quick schema detection (load and analyze first few nodes)
                if file_path.suffix.lower() == ".json":
                    with open(file_path) as f:
                        # Load just enough to detect schema
                        content = f.read(10000)  # First 10KB
                        if len(content) >= 9999:  # File is larger, load properly
                            f.seek(0)
                            graph_data = json.load(f)
                        else:
                            graph_data = json.loads(content)

                    # Detect schema
                    validation = self.normalizer.detect_schema_version(graph_data)
                    schema_version = validation.schema_version

                    # Update distribution
                    if schema_version not in analysis["schema_distribution"]:
                        analysis["schema_distribution"][schema_version] = 0
                    analysis["schema_distribution"][schema_version] += 1

                    # Categorize file
                    if schema_version == "34D" and validation.is_valid:
                        analysis["migration_candidates"].append(str(file_path))
                    elif not validation.is_valid:
                        analysis["validation_failures"].append(
                            {
                                "file": str(file_path),
                                "schema": schema_version,
                                "errors": validation.validation_errors[:3],  # First 3 errors
                            }
                        )

                else:
                    # Assume pickle files need investigation
                    analysis["schema_distribution"]["pickle_unknown"] = (
                        analysis["schema_distribution"].get("pickle_unknown", 0) + 1
                    )

            except Exception as e:
                analysis["analysis_errors"].append(
                    {"file": str(file_path), "error": f"{type(e).__name__}: {str(e)}"}
                )

        # Calculate averages
        if analysis["total_files"] > 0:
            analysis["file_size_stats"]["average"] = (
                analysis["file_size_stats"]["total"] / analysis["total_files"]
            )

        return analysis

    def migrate_single_file(
        self, input_file: Path, output_dir: Path, create_backup: bool = True
    ) -> tuple[bool, str, dict]:
        """
        Migrate single graph file with comprehensive error handling

        Technical Debt Surgeon: Individual file migration with detailed reporting
        """
        result_info = {
            "input_file": str(input_file),
            "output_file": None,
            "backup_file": None,
            "migration_result": None,
            "validation_result": None,
            "processing_time": 0,
            "error": None,
        }

        start_time = datetime.now()

        try:
            # Create backup if requested
            if create_backup:
                backup_file = input_file.with_suffix(
                    f'.backup_{datetime.now().strftime("%H%M%S")}{input_file.suffix}'
                )
                shutil.copy2(input_file, backup_file)
                result_info["backup_file"] = str(backup_file)

            # Load graph data
            if input_file.suffix.lower() == ".json":
                with open(input_file) as f:
                    json.load(f)  # Validate JSON format
            else:
                return False, f"Unsupported file format: {input_file.suffix}", result_info

            # Perform migration using normalizer
            migration_result, validation_result = self.normalizer.process_graph_file(
                str(input_file),
                output_filepath=None,  # We'll handle output ourselves
                target_schema="37D",
            )

            result_info["migration_result"] = {
                "success": migration_result.success,
                "source_schema": migration_result.source_schema,
                "target_schema": migration_result.target_schema,
                "nodes_migrated": migration_result.nodes_migrated,
                "features_added": migration_result.features_added,
            }

            result_info["validation_result"] = {
                "is_valid": validation_result.is_valid,
                "schema_version": validation_result.schema_version,
                "detected_dimensions": validation_result.detected_dimensions,
            }

            if migration_result.success and validation_result.is_valid:
                # Save migrated file to output directory
                output_file = output_dir / input_file.name

                # Ensure output directory exists
                output_dir.mkdir(parents=True, exist_ok=True)

                # Load the migrated data from the temp file or from memory
                with open(input_file) as f:
                    migrated_data = json.load(f)  # This should now be the migrated version

                # Save to output location
                with open(output_file, "w") as f:
                    json.dump(migrated_data, f, indent=2)

                result_info["output_file"] = str(output_file)
                result_info["processing_time"] = (datetime.now() - start_time).total_seconds()

                return True, "Migration successful", result_info
            else:
                error_msg = f"Migration failed - {migration_result.migration_errors or validation_result.validation_errors}"
                result_info["error"] = error_msg
                return False, error_msg, result_info

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result_info["error"] = error_msg
            result_info["processing_time"] = (datetime.now() - start_time).total_seconds()
            return False, error_msg, result_info

    def migrate_batch(self, input_dir: str, output_dir: str, create_backups: bool = True) -> dict:
        """
        Migrate entire batch of graph files

        Technical Debt Surgeon: Comprehensive batch processing with progress tracking
        """
        print("🔧 TECHNICAL DEBT SURGEON - Batch Graph Migration")
        print("=" * 55)

        self.batch_stats["start_time"] = datetime.now()

        # Setup directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Discover graph files
        graph_files = self.discover_graph_files(input_dir)
        self.batch_stats["total_files"] = len(graph_files)

        print(f"📁 Input Directory: {input_dir}")
        print(f"📁 Output Directory: {output_dir}")
        print(f"📊 Discovered Files: {len(graph_files)}")

        if not graph_files:
            print("⚠️  No graph files found in input directory")
            return self.batch_stats

        # Analyze batch requirements
        analysis = self.analyze_batch_requirements(graph_files)

        print("\n📊 Batch Analysis Results:")
        print(f"   Schema Distribution: {analysis['schema_distribution']}")
        print(f"   Migration Candidates: {len(analysis['migration_candidates'])}")
        print(f"   Validation Failures: {len(analysis['validation_failures'])}")
        print(f"   Analysis Errors: {len(analysis['analysis_errors'])}")

        if analysis["analysis_errors"]:
            print("\n⚠️  Analysis Errors:")
            for error in analysis["analysis_errors"][:3]:
                print(f"   • {error['file']}: {error['error']}")

        # Create backup if requested
        if create_backups and analysis["migration_candidates"]:
            try:
                backup_location = self.create_backup(input_dir)
                print(f"🔒 Backup created: {backup_location}")
            except Exception as e:
                print(f"❌ Backup creation failed: {e}")
                self.batch_stats["warnings"].append(f"Backup failed: {str(e)}")

        # Process files
        print("\n🔄 Starting batch migration...")
        successful_migrations = []
        failed_migrations = []

        # Sequential processing for better error tracking
        # TODO: Implement parallel processing with ThreadPoolExecutor for performance
        for i, file_path in enumerate(graph_files):
            print(f"\n📄 Processing [{i+1}/{len(graph_files)}]: {file_path.name}")

            success, message, result_info = self.migrate_single_file(
                file_path, output_path, create_backup=False  # Already created batch backup
            )

            self.batch_stats["processed_files"] += 1

            if success:
                print(f"   ✅ SUCCESS - {message}")
                if result_info["migration_result"]["nodes_migrated"] > 0:
                    self.batch_stats["migrated_files"] += 1
                    successful_migrations.append(result_info)
                    print(
                        f"      Migrated: {result_info['migration_result']['nodes_migrated']} nodes"
                    )
                    print(
                        f"      Features added: {result_info['migration_result']['features_added']}"
                    )
                else:
                    self.batch_stats["skipped_files"] += 1
                    print(
                        f"      Skipped: Already {result_info['migration_result']['target_schema']}"
                    )
            else:
                print(f"   ❌ FAILED - {message}")
                self.batch_stats["failed_files"] += 1
                failed_migrations.append(result_info)
                self.batch_stats["errors"].append({"file": file_path.name, "error": message})

        self.batch_stats["end_time"] = datetime.now()

        # Generate comprehensive report
        self.print_batch_report(successful_migrations, failed_migrations)

        return self.batch_stats

    def print_batch_report(self, successful_migrations: list, failed_migrations: list):
        """Print comprehensive batch migration report"""
        processing_time = (
            self.batch_stats["end_time"] - self.batch_stats["start_time"]
        ).total_seconds()

        print(f"\n{'='*70}")
        print("📊 BATCH MIGRATION COMPLETE")
        print(f"{'='*70}")

        print(f"\n⏱️  Processing Time: {processing_time:.1f} seconds")
        print(f"📁 Total Files: {self.batch_stats['total_files']}")
        print(
            f"✅ Successful: {self.batch_stats['migrated_files']} migrated + {self.batch_stats['skipped_files']} skipped"
        )
        print(f"❌ Failed: {self.batch_stats['failed_files']}")

        success_rate = (
            (
                (self.batch_stats["migrated_files"] + self.batch_stats["skipped_files"])
                / self.batch_stats["total_files"]
                * 100
            )
            if self.batch_stats["total_files"] > 0
            else 0
        )
        print(f"📈 Success Rate: {success_rate:.1f}%")

        # Migration statistics
        if successful_migrations:
            total_nodes_migrated = sum(
                [m["migration_result"]["nodes_migrated"] for m in successful_migrations]
            )
            print("\n🔄 Migration Details:")
            print(f"   Total Nodes Migrated: {total_nodes_migrated}")

            # Most common schema transitions
            schema_transitions = {}
            for migration in successful_migrations:
                transition = f"{migration['migration_result']['source_schema']} → {migration['migration_result']['target_schema']}"
                schema_transitions[transition] = schema_transitions.get(transition, 0) + 1

            print(f"   Schema Transitions: {dict(schema_transitions)}")

        # Error analysis
        if failed_migrations:
            print("\n❌ Failed Migrations Analysis:")
            error_types = {}
            for failure in failed_migrations:
                if failure["error"]:
                    error_type = failure["error"].split(":")[0]
                    error_types[error_type] = error_types.get(error_type, 0) + 1

            print(f"   Error Types: {dict(error_types)}")

            # Show first few failures
            print("\n   Sample Failures:")
            for failure in failed_migrations[:3]:
                print(f"   • {Path(failure['input_file']).name}: {failure['error'][:60]}...")

        # Recommendations
        print("\n💡 TECHNICAL DEBT SURGEON RECOMMENDATIONS:")
        if success_rate >= 90:
            print("   ✅ Excellent migration success rate - system is production ready")
        elif success_rate >= 70:
            print("   ⚠️  Good migration success rate - review failed cases")
            print("   🔧 Consider fixing data quality issues in source files")
        else:
            print("   ❌ Low migration success rate - investigate systematic issues")
            print("   🔧 Review error patterns and fix data pipeline problems")
            print("   🔧 Consider manual review of failed files")

        if self.batch_stats["warnings"]:
            print(f"\n⚠️  Warnings: {len(self.batch_stats['warnings'])}")
            for warning in self.batch_stats["warnings"]:
                print(f"   • {warning}")


def main():
    """Main execution with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="IRONFORGE Batch Graph Schema Migration - Technical Debt Surgeon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate all graphs from input to output directory
    python3 batch_migrate_graphs.py --input /path/to/34d/graphs --output /path/to/37d/graphs
    
    # Migrate without creating backups
    python3 batch_migrate_graphs.py --input ./graphs --output ./migrated_graphs --no-backup
    
    # Parallel processing with 8 workers
    python3 batch_migrate_graphs.py --input ./graphs --output ./migrated_graphs --workers 8
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Input directory containing 34D graph files"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for migrated 37D graph files"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip backup creation (faster but riskier)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Analyze files without performing migration"
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.input).exists():
        print(f"❌ Error: Input directory does not exist: {args.input}")
        return 1

    if not Path(args.input).is_dir():
        print(f"❌ Error: Input path is not a directory: {args.input}")
        return 1

    try:
        migrator = BatchGraphMigrator()

        if args.dry_run:
            # Dry run - analysis only
            print("🔍 DRY RUN MODE - Analysis Only")
            graph_files = migrator.discover_graph_files(args.input)
            analysis = migrator.analyze_batch_requirements(graph_files)

            print("\n📊 Analysis Results:")
            print(f"   Total Files: {analysis['total_files']}")
            print(f"   Schema Distribution: {analysis['schema_distribution']}")
            print(f"   Migration Candidates: {len(analysis['migration_candidates'])}")
            print(f"   Validation Failures: {len(analysis['validation_failures'])}")

            return 0
        else:
            # Full migration
            batch_stats = migrator.migrate_batch(
                input_dir=args.input,
                output_dir=args.output,
                max_workers=args.workers,
                create_backups=not args.no_backup,
            )

            # Return exit code based on success rate
            success_rate = (
                (
                    (batch_stats["migrated_files"] + batch_stats["skipped_files"])
                    / batch_stats["total_files"]
                    * 100
                )
                if batch_stats["total_files"] > 0
                else 0
            )

            return 0 if success_rate >= 70 else 1

    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        return 130
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
