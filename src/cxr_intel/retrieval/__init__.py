"""Retrieval backends. Common protocol allows swapping ColPali / BiomedCLIP."""
from cxr_intel.retrieval.base import Retriever, RetrievalHit

__all__ = ["Retriever", "RetrievalHit"]
