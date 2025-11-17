"""
Relation Extractor
Extracts relationships between financial entities
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import spacy


@dataclass
class Relation:
    """Data class for entity relations"""
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    confidence: float
    context: str
    
    def __str__(self):
        return f"{self.subject} --[{self.relation}]--> {self.object}"


class RelationExtractor:
    """
    Extract semantic relationships between entities
    """
    
    def __init__(self):
        """Initialize relation extractor"""
        print("Initializing Relation Extractor...")
        
        # Load spaCy for dependency parsing
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.use_spacy = True
        except:
            print("⚠ spaCy not available, using pattern-based only")
            self.use_spacy = False
        
        # Define relation patterns
        self.relation_patterns = self._init_patterns()
        
        print("✓ Relation Extractor initialized")
    
    def _init_patterns(self) -> List[Dict]:
        """Initialize relation extraction patterns"""
        return [
            # Company reported/announced metric
            {
                'name': 'REPORTED',
                'pattern': r'([\w\s]+(?:Inc\.|Corp\.|Ltd\.|LLC)?)\s+(reported|announced|posted|disclosed|generated)\s+([^.]{1,100})',
                'subject_type': 'COMPANY',
                'object_type': 'METRIC',
                'confidence': 0.9
            },
            # Metric increased/decreased by amount
            {
                'name': 'CHANGED_BY',
                'pattern': r'(revenue|profit|income|earnings|sales|eps)\s+(increased|decreased|grew|fell|rose|dropped)\s+(?:by\s+)?([\d.]+%)',
                'subject_type': 'METRIC',
                'object_type': 'PERCENTAGE',
                'confidence': 0.95
            },
            # Company acquired/bought company
            {
                'name': 'ACQUIRED',
                'pattern': r'([\w\s]+)\s+(acquired|bought|purchased)\s+([\w\s]+(?:Inc\.|Corp\.|Ltd\.)?)',
                'subject_type': 'COMPANY',
                'object_type': 'COMPANY',
                'confidence': 0.9
            },
            # Person role at company
            {
                'name': 'ROLE_AT',
                'pattern': r'((?:CEO|CFO|President|Chairman|Director)\s+[\w\s]+?)\s+(?:of|at)\s+([\w\s]+(?:Inc\.|Corp\.|Ltd\.)?)',
                'subject_type': 'PERSON',
                'object_type': 'COMPANY',
                'confidence': 0.85
            },
            # Person stated/said
            {
                'name': 'STATED',
                'pattern': r'([\w\s]+?)\s+(stated|said|commented|noted|announced)\s+that\s+([^.]{1,150})',
                'subject_type': 'PERSON',
                'object_type': 'STATEMENT',
                'confidence': 0.8
            },
            # Company has metric of value
            {
                'name': 'HAS_VALUE',
                'pattern': r'([\w\s]+)\s+(?:has|had|reported|posted)\s+(revenue|profit|income|sales|assets)\s+of\s+(\$[\d,.]+\s?[BMK]?)',
                'subject_type': 'COMPANY',
                'object_type': 'MONEY',
                'confidence': 0.9
            },
            # Company operates in location
            {
                'name': 'OPERATES_IN',
                'pattern': r'([\w\s]+(?:Inc\.|Corp\.|Ltd\.)?)\s+(?:operates|has presence|does business)\s+in\s+([\w\s]+)',
                'subject_type': 'COMPANY',
                'object_type': 'LOCATION',
                'confidence': 0.7
            },
            # Metric attributable to segment
            {
                'name': 'ATTRIBUTABLE_TO',
                'pattern': r'(revenue|sales|income|profit)\s+(?:from|of|attributable to)\s+([\w\s]+)',
                'subject_type': 'METRIC',
                'object_type': 'SEGMENT',
                'confidence': 0.8
            },
            # Company competes with company
            {
                'name': 'COMPETES_WITH',
                'pattern': r'([\w\s]+)\s+(?:competes with|rivals|competitor of)\s+([\w\s]+)',
                'subject_type': 'COMPANY',
                'object_type': 'COMPANY',
                'confidence': 0.75
            },
            # Comparison relations
            {
                'name': 'COMPARED_TO',
                'pattern': r'([\w\s]+)\s+(?:compared to|versus|vs\.?)\s+([\w\s]+)',
                'subject_type': 'ANY',
                'object_type': 'ANY',
                'confidence': 0.8
            }
        ]
    
    def extract_relations(self, text: str, 
                         entities: Optional[List[Dict]] = None) -> List[Relation]:
        """
        Extract all relations from text
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities for context
            
        Returns:
            List of Relation objects
        """
        relations = []
        
        # Pattern-based extraction
        pattern_relations = self._extract_pattern_relations(text)
        relations.extend(pattern_relations)
        
        # Dependency-based extraction (if spaCy available)
        if self.use_spacy:
            dep_relations = self._extract_dependency_relations(text)
            relations.extend(dep_relations)
        
        # Deduplicate
        relations = self._deduplicate_relations(relations)
        
        return relations
    
    def _extract_pattern_relations(self, text: str) -> List[Relation]:
        """Extract relations using regex patterns"""
        relations = []
        
        for pattern_config in self.relation_patterns:
            matches = re.finditer(pattern_config['pattern'], text, re.IGNORECASE)
            
            for match in matches:
                try:
                    subject = match.group(1).strip()
                    relation_type = pattern_config['name']
                    obj = match.group(3).strip() if match.lastindex >= 3 else match.group(2).strip()
                    
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    relation = Relation(
                        subject=subject,
                        subject_type=pattern_config['subject_type'],
                        relation=relation_type,
                        object=obj,
                        object_type=pattern_config['object_type'],
                        confidence=pattern_config['confidence'],
                        context=context
                    )
                    
                    relations.append(relation)
                    
                except (IndexError, AttributeError):
                    continue
        
        return relations
    
    def _extract_dependency_relations(self, text: str) -> List[Relation]:
        """Extract relations using dependency parsing"""
        relations = []
        
        if not self.use_spacy:
            return relations
        
        doc = self.nlp(text)
        
        # Extract subject-verb-object triples
        for sent in doc.sents:
            for token in sent:
                # Look for verb tokens
                if token.pos_ == 'VERB':
                    # Find subject
                    subject = None
                    for child in token.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            subject = self._get_full_phrase(child)
                            break
                    
                    # Find object
                    obj = None
                    for child in token.children:
                        if child.dep_ in ['dobj', 'attr', 'pobj']:
                            obj = self._get_full_phrase(child)
                            break
                    
                    # If we have both subject and object, create relation
                    if subject and obj and len(subject) > 2 and len(obj) > 2:
                        relation = Relation(
                            subject=subject,
                            subject_type='ENTITY',
                            relation=token.lemma_.upper(),
                            object=obj,
                            object_type='ENTITY',
                            confidence=0.7,
                            context=sent.text
                        )
                        relations.append(relation)
        
        return relations
    
    def _get_full_phrase(self, token) -> str:
        """Get full phrase from token including modifiers"""
        # Get all children recursively
        phrase_tokens = [token]
        
        for child in token.children:
            if child.dep_ in ['compound', 'amod', 'det', 'poss']:
                phrase_tokens.append(child)
        
        # Sort by position and join
        phrase_tokens.sort(key=lambda t: t.i)
        return ' '.join([t.text for t in phrase_tokens])
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate or highly similar relations"""
        if not relations:
            return []
        
        # Sort by confidence
        relations.sort(key=lambda r: r.confidence, reverse=True)
        
        unique_relations = []
        seen_triples = set()
        
        for relation in relations:
            # Create a normalized representation
            triple = (
                relation.subject.lower().strip(),
                relation.relation.lower(),
                relation.object.lower().strip()
            )
            
            if triple not in seen_triples:
                unique_relations.append(relation)
                seen_triples.add(triple)
        
        return unique_relations
    
    def extract_company_relations(self, text: str,
                                  company_name: str) -> List[Relation]:
        """Extract all relations involving a specific company"""
        all_relations = self.extract_relations(text)
        
        company_lower = company_name.lower()
        return [
            r for r in all_relations 
            if company_lower in r.subject.lower() or company_lower in r.object.lower()
        ]
    
    def build_knowledge_graph(self, text: str) -> Dict:
        """
        Build a knowledge graph from extracted relations
        
        Returns:
            Dictionary representing graph structure
        """
        relations = self.extract_relations(text)
        
        # Build graph structure
        graph = {
            'nodes': set(),
            'edges': [],
            'node_types': {}
        }
        
        for relation in relations:
            # Add nodes
            graph['nodes'].add(relation.subject)
            graph['nodes'].add(relation.object)
            
            # Add node types
            graph['node_types'][relation.subject] = relation.subject_type
            graph['node_types'][relation.object] = relation.object_type
            
            # Add edge
            graph['edges'].append({
                'source': relation.subject,
                'target': relation.object,
                'relation': relation.relation,
                'confidence': relation.confidence
            })
        
        # Convert nodes set to list for JSON serialization
        graph['nodes'] = list(graph['nodes'])
        
        return graph
    
    def get_relation_statistics(self, text: str) -> Dict:
        """Get statistics about extracted relations"""
        relations = self.extract_relations(text)
        
        # Count by relation type
        relation_counts = {}
        for rel in relations:
            relation_counts[rel.relation] = relation_counts.get(rel.relation, 0) + 1
        
        # Count by subject type
        subject_type_counts = {}
        for rel in relations:
            subject_type_counts[rel.subject_type] = subject_type_counts.get(rel.subject_type, 0) + 1
        
        return {
            'total_relations': len(relations),
            'unique_subjects': len(set(r.subject for r in relations)),
            'unique_objects': len(set(r.object for r in relations)),
            'by_relation_type': relation_counts,
            'by_subject_type': subject_type_counts,
            'avg_confidence': sum(r.confidence for r in relations) / len(relations) if relations else 0
        }
    
    def visualize_relations(self, text: str, max_relations: int = 10) -> str:
        """Create text-based visualization of relations"""
        relations = self.extract_relations(text)
        
        # Sort by confidence
        relations = sorted(relations, key=lambda r: r.confidence, reverse=True)
        relations = relations[:max_relations]
        
        output = ["\n" + "="*60]
        output.append("EXTRACTED RELATIONS")
        output.append("="*60 + "\n")
        
        for i, rel in enumerate(relations, 1):
            output.append(f"{i}. {rel.subject}")
            output.append(f"   --[{rel.relation}]-->")
            output.append(f"   {rel.object}")
            output.append(f"   (confidence: {rel.confidence:.2f})")
            output.append("")
        
        return "\n".join(output)


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = RelationExtractor()
    
    # Sample text
    sample_text = """
    Apple Inc. reported Q3 2024 revenue of $90.1 billion, up 15.2% year-over-year.
    CEO Tim Cook stated that iPhone sales grew 12% to $45.8 billion. The company
    operates in over 175 countries and competes with Samsung and Google in various
    markets. Net income increased by 8% compared to the previous quarter.
    """
    
    print("\nExtracting relations...")
    relations = extractor.extract_relations(sample_text)
    
    print(f"\n✓ Found {len(relations)} relations")
    
    # Show relations
    print(extractor.visualize_relations(sample_text, max_relations=8))
    
    # Show statistics
    stats = extractor.get_relation_statistics(sample_text)
    print("\nRelation Statistics:")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Unique subjects: {stats['unique_subjects']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    
    # Build knowledge graph
    graph = extractor.build_knowledge_graph(sample_text)
    print(f"\nKnowledge Graph:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['edges'])}")
    
    print("\n✓ Relation Extractor Module Ready!")  