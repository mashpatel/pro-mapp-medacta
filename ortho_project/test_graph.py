import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def test_connection():
    logger.info(f"Attempting to connect to: {NEO4J_URI}")
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            logger.info("Driver created successfully")
            with driver.session() as session:
                logger.info("Session created successfully")
                result = session.run("RETURN 1 AS num")
                record = result.single()
                logger.info(f"Query executed successfully. Result: {record['num']}")
                print(f"Connection successful. Result: {record['num']}")
    except Exception as e:
        logger.error(f"Connection failed. Error: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_connection()