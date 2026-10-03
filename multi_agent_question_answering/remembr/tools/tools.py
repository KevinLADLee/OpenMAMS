# Import things that are needed generically
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool, tool
import numpy as np
from time import strftime, localtime
import time, datetime

from typing import Any, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document

### Doc formatting for the last LLM
def format_document(docs, ref_time=None):
    out_string = ""
    for doc in docs:
        if len(doc.metadata['time']) == 2:
            t = doc.metadata['time'][0]
        else:
            t = doc.metadata['time']
        
        if ref_time:
            t += ref_time
        t = localtime(t)
        t = strftime('%Y-%m-%d %H:%M:%S', t)

        s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}. "
        s += f"The robot saw the following: {doc.page_content}\n\n"
        out_string += s
    return out_string

def format_docs(docs):
    out_string = ""
    for doc in docs:
        if len(doc['time']) == 2:
            t = doc['time'][0]
        else:
            t = doc['time']

        t = localtime(t)
        t = strftime('%Y-%m-%d %H:%M:%S', t)

        s = f"At time={t}, the robot was at an average position of {np.array(doc['position']).round(3).tolist()}. "
        s += f"The robot saw the following: {doc['caption']}\n\n"
        out_string += s
    return out_string




def search_by_position(pos_db, ref_time, query: tuple) -> str:
    # docs = pos_db.similarity_search_by_vector(np.array(query))
    docs = similarity_search_with_score_by_vector(pos_db, np.array(query).astype(float))

    docs = format_document(docs, ref_time=ref_time)

    """Look up things online."""
    return docs

def search_by_time(time_db, ref_time, hms_time: str) -> str:
    print("Searching by time", hms_time)
    # Input is time like 08:20:30
    # need to convert to searchable time
    t = localtime(ref_time)
    mdy_date = strftime('%m/%d/%Y', t)
    template = "%m/%d/%Y %H:%M:%S"

    # if the hms_time is already in the mdy hms format without me doing anything, let's just use that.
    # bad llms don't listen :(
    try:
        res = bool(datetime.strptime(query, template))
    except ValueError:
        res = False

    hms_time = hms_time.strip()
    if not res: # convert to the right format then
        hms_time = mdy_date + ' ' + hms_time

    query = time.mktime(datetime.datetime.strptime(hms_time,template).timetuple()) - ref_time
    # convert from hms_time to something searchable


    docs = similarity_search_with_score_by_vector(time_db, np.array([query, 0]))

    docs = format_document(docs, ref_time=ref_time)
    # np.unique([doc.metadata['time'][0] for doc in docs])
    """Look up things online."""
    return docs



def search_by_text(retriever, ref_time, query: str) -> str:

    docs = retriever.invoke(query)

    docs = format_document(docs, ref_time=ref_time)

    """Look up things online."""
    return docs


# NOTE: This version of the code returns the vector
def similarity_search_with_score_by_vector(
        pos_db,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if pos_db.col is None:
            print("No existing collection to search.")
            return []

        if param is None:
            param = pos_db.search_params

        # Determine result metadata fields with PK
        output_fields = pos_db.fields[:]

        # Fix: Explicitly add vector_field to output_fields if not present
        # Milvus needs vector_field in output_fields to return it
        if pos_db._vector_field not in output_fields:
            output_fields.append(pos_db._vector_field)

        timeout = pos_db.timeout or timeout
        # Perform the search.
        res = pos_db.col.search(
            data=[embedding],
            anns_field=pos_db._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results
        ret = []
        for result in res[0]:
            # Fix: Use result.entity.fields instead of result.entity.get()
            # result.entity.get() doesn't properly return vector fields
            if hasattr(result.entity, 'fields'):
                data = {x: result.entity.fields.get(x) for x in output_fields}
            else:
                data = {x: result.entity.get(x) for x in output_fields}

            # IMPORTANT: Save vector_field value BEFORE calling _parse_document
            # _parse_document will remove vector_field from the data dict
            vector_field_name = pos_db._vector_field
            vector_field_value = data.get(vector_field_name)

            doc = pos_db._parse_document(data)

            # Fix: Add vector_field back to metadata after parsing
            # langchain_milvus._parse_document removes vector_field from metadata,
            # but downstream code expects it (e.g., format_document accessing position/time)
            if vector_field_value is not None and vector_field_name not in doc.metadata:
                doc.metadata[vector_field_name] = vector_field_value

            pair = (doc, result.score)
            ret.append(pair)

        return [doc for doc, _ in ret]