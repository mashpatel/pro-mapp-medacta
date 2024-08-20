import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import sqlite3
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# SQLite database file
SQLITE_DB = "medacta.db"

def remove_constraints(driver):
    with driver.session() as session:
        session.run("CALL apoc.schema.assert({}, {})")
        logger.info("All constraints have been removed.")

def get_data_from_sqlite():
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM surgical_data")
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    
    conn.close()
    return columns, rows

def clean_props(props):
    return {k: str(v) for k, v in props.items() if v is not None and v != ''}

def process_record(record):
    def clean_props(props):
        return {k: str(v) for k, v in props.items() if v is not None and v != ''}

    processed = {
        'PathwayPID': record['PathwayPID'],
        'location_props': clean_props({
            'Location_txt': record.get('Location_txt'),
        }),
        'surgical_team_props': clean_props({
            'Location_txt': record.get('Location_txt'),  # Used as identifier for SurgicalTeam
            'Clinician_txt': record.get('Clinician_txt'),
            'Assistant': record.get('Assistant'),
            'FellowUsed_txt': record.get('FellowUsed_txt'),
        })
    }

    node_props = {
        'patient_props': clean_props({
            'Age': record.get('Age'),
            'Gender_txt': record.get('Gender_txt'),
            'Height': record.get('Height'),
            'Weight': record.get('Weight'),
            'BMI': record.get('BMI'),
            'ObesityComplexityModifier': record.get('ObesityComplexityModifier'),
            'Sensitin': record.get('Sensitin')
        }),
        'procedure_props': clean_props({
            'Procedure_txt': record.get('Procedure_txt'),
            'Side_txt': record.get('Side_txt'),
            'Created': record.get('Created'),
            'Complete': record.get('Complete'),
            'CompleteBy_txt': record.get('CompleteBy_txt'),
            'CompleteDate': record.get('CompleteDate'),
            'ChangedSurgicalPlan': record.get('ChangedSurgicalPlan'),
            'EstimatedBloodLoss_txt': record.get('EstimatedBloodLoss_txt'),
            'TourniquetTime': record.get('TourniquetTime'),
            'IntraOpXrays_txt': record.get('IntraOpXrays_txt'),
            'HardwareRemoved': record.get('HardwareRemoved'),
            'AdditionalTibialVVRecut_txt': record.get('AdditionalTibialVVRecut_txt'),
            'AddNotes': record.get('AddNotes')
        }),
        'pathway_props': clean_props({
            'AdmitScore': record.get('AdmitScore'),
            'DischargeDate': record.get('DischargeDate'),
            'Baseline': record.get('Baseline')
        }),
        'knee_props': clean_props({
            'OADeformity_txt': record.get('OADeformity_txt'),
            'PreOperativeAlignment': record.get('PreOperativeAlignment'),
            'PostOperativeAlignment': record.get('PostOperativeAlignment'),
            'Alignment_txt': record.get('Alignment_txt'),
            'PCLRelease': record.get('PCLRelease'),
            'LateralRetinacularRelease_txt': record.get('LateralRetinacularRelease_txt'),
            'ValgusRelease': record.get('ValgusRelease'),
            'VarusRelease': record.get('VarusRelease')
        }),
        'implant_props': clean_props({
            'FemoralComponent_txt': record.get('FemoralComponent_txt'),
            'FemoralSize_txt': record.get('FemoralSize_txt'),
            'TibialComponent': record.get('TibialComponent'),
            'TibialSize': record.get('TibialSize'),
            'PatellaComponent_txt': record.get('PatellaComponent_txt'),
            'PatellaSize': record.get('PatellaSize'),
            'Poly': record.get('Poly'),
            'PolySize': record.get('PolySize'),
            'PolyThickness': record.get('PolyThickness'),
            'FemoroTibialLinkage': record.get('FemoroTibialLinkage'),
            'TibialStemExtension': record.get('TibialStemExtension'),
            'BoneCementTypeFix_txt': record.get('BoneCementTypeFix_txt')
        }),
        'femoral_resection_props': clean_props({
            'DFRLateralCondyleInitialThickness_txt': record.get('DFRLateralCondyleInitialThickness_txt'),
            'DFRLateralCondyleFinalThickness_txt': record.get('DFRLateralCondyleFinalThickness_txt'),
            'DFRLateralCondyleStatus': record.get('DFRLateralCondyleStatus'),
            'DFRLateralCondyleStatus_txt': record.get('DFRLateralCondyleStatus_txt'),
            'DFRLateralCondyleRecut_txt': record.get('DFRLateralCondyleRecut_txt'),
            'DFRLCRecutAmount': record.get('DFRLCRecutAmount'),
            'DFRLateralCondyleWasher_txt': record.get('DFRLateralCondyleWasher_txt'),
            'DFRLCWasherAmount': record.get('DFRLCWasherAmount'),
            'DFRMedialCondyleInitialThickness_txt': record.get('DFRMedialCondyleInitialThickness_txt'),
            'DFRMedialCondyleFinalThickness_txt': record.get('DFRMedialCondyleFinalThickness_txt'),
            'DFRMedialCondyleStatus_txt': record.get('DFRMedialCondyleStatus_txt'),
            'DFRMedialCondyleRecut_txt': record.get('DFRMedialCondyleRecut_txt'),
            'DFRMCRecutAmount': record.get('DFRMCRecutAmount'),
            'DFRMedialCondyleWasher_txt': record.get('DFRMedialCondyleWasher_txt'),
            'DFRMCWasherAmount': record.get('DFRMCWasherAmount'),
            'PFRLateralCondyleInitialThickness_txt': record.get('PFRLateralCondyleInitialThickness_txt'),
            'PFRLateralCondyleFinalThickness_txt': record.get('PFRLateralCondyleFinalThickness_txt'),
            'PFRMedialCondyleInitialThickness_txt': record.get('PFRMedialCondyleInitialThickness_txt'),
            'PFRMedialCondyleFinalThickness_txt': record.get('PFRMedialCondyleFinalThickness_txt'),
            'PFLCRecut': record.get('PFLCRecut')
        }),
        'tibial_resection_props': clean_props({
            'TibialResection': record.get('TibialResection'),
            'TibialResectionPosteriorSlope': record.get('TibialResectionPosteriorSlope'),
            'TRLateralInitialThickness': record.get('TRLateralInitialThickness'),
            'TRLateralFinalThickness': record.get('TRLateralFinalThickness'),
            'TRLateralRecutAmount': record.get('TRLateralRecutAmount'),
            'TRMedialInitialThickness': record.get('TRMedialInitialThickness'),
            'TRMedialFinalThickness': record.get('TRMedialFinalThickness'),
            'TRMedialRecutAmount': record.get('TRMedialRecutAmount')
        }),
        'anesthesia_props': clean_props({
            'Anesthesia': record.get('Anesthesia'),
            'AnesthesiaExemption_txt': record.get('AnesthesiaExemption_txt'),
            'AnesthesiaO1_txt': record.get('AnesthesiaO1_txt'),
            'AnesthesiaO2_txt': record.get('AnesthesiaO2_txt'),
            'AnesthesiaO3_txt': record.get('AnesthesiaO3_txt'),
            'PreOpPainBlock': record.get('PreOpPainBlock')
        }),
        'medication_props': clean_props({
            'IntraoperativeMeds': record.get('IntraoperativeMeds'),
            'IntraoperativeMedsO1_txt': record.get('IntraoperativeMedsO1_txt'),
            'IntraoperativeMedsO2_txt': record.get('IntraoperativeMedsO2_txt'),
            'LactatedRingers': record.get('LactatedRingers'),
            'VancomycinDosage': record.get('VancomycinDosage')
        }),
        'diagnosis_props': clean_props({
            'PreOpDiagnosis': record.get('PreOpDiagnosis'),
            'PostOpDiagnosis': record.get('PostOpDiagnosis')
        }),
        'balance_props': clean_props({
            'Goniometric10mmThickness': record.get('Goniometric10mmThickness'),
            'Goniometric12mmThickness': record.get('Goniometric12mmThickness'),
            'Goniometric14mmThickness': record.get('Goniometric14mmThickness'),
            'Goniometric16mmThickness': record.get('Goniometric16mmThickness'),
            'Goniometric18mmThickness': record.get('Goniometric18mmThickness'),
            'Goniometric20mmThickness': record.get('Goniometric20mmThickness'),
            'Ticks10mm': record.get('Ticks10mm'),
            'Ticks12mm': record.get('Ticks12mm'),
            'Ticks14mm': record.get('Ticks14mm')
        })
    }

    processed.update({k: v for k, v in node_props.items() if v})
    return processed

def create_nodes_and_relationships(tx, records):
    query = """
    UNWIND $records AS record
    
    MERGE (patient:Patient {PathwayPID: record.PathwayPID})
    SET patient += CASE WHEN record.patient_props IS NOT NULL THEN record.patient_props ELSE {} END
    
    MERGE (procedure:Procedure {PathwayPID: record.PathwayPID})
    SET procedure += CASE WHEN record.procedure_props IS NOT NULL THEN record.procedure_props ELSE {} END
    
    MERGE (pathway:Pathway {PathwayPID: record.PathwayPID})
    SET pathway += CASE WHEN record.pathway_props IS NOT NULL THEN record.pathway_props ELSE {} END
    
    MERGE (knee:Knee {PathwayPID: record.PathwayPID})
    SET knee += CASE WHEN record.knee_props IS NOT NULL THEN record.knee_props ELSE {} END
    
    MERGE (implant:Implant {PathwayPID: record.PathwayPID})
    SET implant += CASE WHEN record.implant_props IS NOT NULL THEN record.implant_props ELSE {} END
    
    MERGE (femoral_resection:FemoralResection {PathwayPID: record.PathwayPID})
    SET femoral_resection += CASE WHEN record.femoral_resection_props IS NOT NULL THEN record.femoral_resection_props ELSE {} END
    
    MERGE (tibial_resection:TibialResection {PathwayPID: record.PathwayPID})
    SET tibial_resection += CASE WHEN record.tibial_resection_props IS NOT NULL THEN record.tibial_resection_props ELSE {} END
    
    MERGE (anesthesia:Anesthesia {PathwayPID: record.PathwayPID})
    SET anesthesia += CASE WHEN record.anesthesia_props IS NOT NULL THEN record.anesthesia_props ELSE {} END
    
    MERGE (medication:Medication {PathwayPID: record.PathwayPID})
    SET medication += CASE WHEN record.medication_props IS NOT NULL THEN record.medication_props ELSE {} END
    
    MERGE (location:Location {Location_txt: record.location_props.Location_txt})
    SET location += record.location_props
    
    MERGE (surgical_team:SurgicalTeam {Location_txt: record.surgical_team_props.Location_txt})
    SET surgical_team += record.surgical_team_props
    
    MERGE (diagnosis:Diagnosis {PathwayPID: record.PathwayPID})
    SET diagnosis += CASE WHEN record.diagnosis_props IS NOT NULL THEN record.diagnosis_props ELSE {} END
    
    MERGE (balance:BalanceMeasurement {PathwayPID: record.PathwayPID})
    SET balance += CASE WHEN record.balance_props IS NOT NULL THEN record.balance_props ELSE {} END
    
    MERGE (patient)-[:UNDERWENT]->(procedure)
    MERGE (patient)-[:FOLLOWS]->(pathway)
    MERGE (procedure)-[:PERFORMED_ON]->(knee)
    MERGE (procedure)-[:USED]->(implant)
    MERGE (procedure)-[:INVOLVED]->(femoral_resection)
    MERGE (procedure)-[:INVOLVED]->(tibial_resection)
    MERGE (procedure)-[:USED]->(anesthesia)
    MERGE (procedure)-[:ADMINISTERED]->(medication)
    MERGE (procedure)-[:PERFORMED_AT]->(location)
    MERGE (location)-[:HAS_TEAM]->(surgical_team)
    MERGE (procedure)-[:HAS_DIAGNOSIS]->(diagnosis)
    MERGE (procedure)-[:MEASURED]->(balance)
    """
    
    result = tx.run(query, records=records)
    return result.consume().counters

def process_records(driver, columns, rows):
    failed_pathways = []
    records = []
    total_processed = 0
    
    for row in tqdm(rows, desc="Processing records"):
        record = dict(zip(columns, row))
        try:
            processed_record = process_record(record)
            records.append(processed_record)
            
            if len(records) >= 100:  # Process in batches of 100
                try:
                    with driver.session() as session:
                        counters = session.execute_write(create_nodes_and_relationships, records)
                        logger.debug(f"Batch processed. Counters: {counters}")
                    total_processed += len(records)
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    failed_pathways.extend([r.get('PathwayPID') for r in records])
                records = []
        except Exception as e:
            logger.error(f"Error processing record for PathwayPID {record.get('PathwayPID')}: {str(e)}")
            failed_pathways.append(record.get('PathwayPID'))
    
    # Process any remaining records
    if records:
        try:
            with driver.session() as session:
                counters = session.execute_write(create_nodes_and_relationships, records)
                logger.debug(f"Final batch processed. Counters: {counters}")
            total_processed += len(records)
        except Exception as e:
            logger.error(f"Error processing final batch: {str(e)}")
            failed_pathways.extend([r.get('PathwayPID') for r in records])
    
    logger.info(f"Total records processed: {total_processed}")
    return failed_pathways

def main():
    logger.info("Starting data loading process")
    logger.info("Fetching data from SQLite database")
    columns, rows = get_data_from_sqlite()
    logger.info(f"Fetched {len(rows)} records from SQLite")

    logger.info("Connecting to Neo4j and processing records")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        remove_constraints(driver)
        failed_pathways = process_records(driver, columns, rows)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        driver.close()
    
    logger.info("Data loading process completed")
    logger.info(f"Number of failed Pathways: {len(failed_pathways)}")
    if failed_pathways:
        logger.info("Failed Pathways:")
        for pathway in failed_pathways[:10]:  # Show first 10 as an example
            logger.info(pathway)
        if len(failed_pathways) > 10:
            logger.info(f"... and {len(failed_pathways) - 10} more")

if __name__ == "__main__":
    main()