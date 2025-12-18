"""
Risk Analyzer
Identify and assess financial risks in documents
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sentiment_model import FinancialSentiment
from extraction.pattern_matcher import PatternMatcher


@dataclass
class Risk:
    """Risk item"""
    category: str
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    context: str
    indicators: List[str]
    
    def __str__(self):
        return f"[{self.severity}] {self.category}: {self.description}"


class RiskAnalyzer:
    """
    Analyze and assess financial risks
    """
    
    def __init__(self):
        """Initialize risk analyzer"""
        print("Initializing Risk Analyzer...")
        
        self.sentiment = FinancialSentiment()
        self.pattern_matcher = PatternMatcher()
        
        # Risk categories and keywords
        self.risk_categories = {
            'MARKET_RISK': {
                'keywords': ['market volatility', 'market conditions', 'economic uncertainty',
                           'competitive pressure', 'market share', 'pricing pressure'],
                'severity_base': 'MEDIUM'
            },
            'OPERATIONAL_RISK': {
                'keywords': ['supply chain', 'production', 'operations', 'disruption',
                           'capacity', 'efficiency', 'quality issues'],
                'severity_base': 'MEDIUM'
            },
            'FINANCIAL_RISK': {
                'keywords': ['liquidity', 'debt', 'cash flow', 'funding', 'credit',
                           'covenant', 'leverage', 'solvency'],
                'severity_base': 'HIGH'
            },
            'REGULATORY_RISK': {
                'keywords': ['regulation', 'compliance', 'legal', 'lawsuit',
                           'investigation', 'fine', 'penalty', 'regulatory'],
                'severity_base': 'HIGH'
            },
            'STRATEGIC_RISK': {
                'keywords': ['strategic', 'competition', 'market position',
                           'technology', 'innovation', 'disruption'],
                'severity_base': 'MEDIUM'
            },
            'CYBERSECURITY_RISK': {
                'keywords': ['cybersecurity', 'data breach', 'security', 'hack',
                           'cyber attack', 'data protection'],
                'severity_base': 'HIGH'
            },
            'REPUTATIONAL_RISK': {
                'keywords': ['reputation', 'brand', 'customer satisfaction',
                           'public perception', 'scandal', 'controversy'],
                'severity_base': 'MEDIUM'
            }
        }
        
        # Severity modifiers
        self.severity_modifiers = {
            'HIGH': ['significant', 'substantial', 'major', 'severe', 'critical',
                    'material', 'adverse', 'negative'],
            'LOW': ['minimal', 'minor', 'limited', 'manageable', 'mitigated']
        }
        
        print("✓ Risk Analyzer initialized")
    
    def analyze_risks(self, text: str) -> List[Risk]:
        """
        Analyze text for financial risks
        
        Args:
            text: Document text
            
        Returns:
            List of identified risks
        """
        risks = []
        
        # Extract risk-related sentences
        risk_sentences = self._extract_risk_sentences(text)
        
        # Analyze each sentence
        for sentence in risk_sentences:
            # Categorize risk
            category = self._categorize_risk(sentence)
            
            if category:
                # Assess severity
                severity = self._assess_severity(sentence, category)
                
                # Extract indicators
                indicators = self._extract_indicators(sentence, category)
                
                # Calculate confidence
                confidence = self._calculate_confidence(sentence, category, indicators)
                
                # Create risk object
                risk = Risk(
                    category=category,
                    description=self._generate_description(sentence),
                    severity=severity,
                    confidence=confidence,
                    context=sentence,
                    indicators=indicators
                )
                
                risks.append(risk)
        
        # Deduplicate similar risks
        risks = self._deduplicate_risks(risks)
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        risks.sort(key=lambda r: severity_order.get(r.severity, 4))
        
        return risks
    
    def _extract_risk_sentences(self, text: str) -> List[str]:
        """Extract sentences that may contain risk information"""
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(text)
        risk_sentences = []
        
        risk_indicators = [
            'risk', 'uncertainty', 'challenge', 'concern', 'threat',
            'vulnerability', 'exposure', 'adverse', 'negative', 'decline',
            'loss', 'failure', 'inability', 'could', 'may', 'might'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in risk_indicators):
                risk_sentences.append(sentence)
        
        return risk_sentences
    
    def _categorize_risk(self, sentence: str) -> Optional[str]:
        """Categorize risk based on keywords"""
        sentence_lower = sentence.lower()
        
        # Find matching category
        matches = []
        for category, config in self.risk_categories.items():
            keyword_count = sum(1 for kw in config['keywords'] if kw in sentence_lower)
            if keyword_count > 0:
                matches.append((category, keyword_count))
        
        if matches:
            # Return category with most keyword matches
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0][0]
        
        return None
    
    def _assess_severity(self, sentence: str, category: str) -> str:
        """Assess severity of risk"""
        base_severity = self.risk_categories[category]['severity_base']
        sentence_lower = sentence.lower()
        
        # Check for severity modifiers
        high_count = sum(1 for mod in self.severity_modifiers['HIGH'] if mod in sentence_lower)
        low_count = sum(1 for mod in self.severity_modifiers['LOW'] if mod in sentence_lower)
        
        if high_count > low_count:
            # Escalate severity
            if base_severity == 'LOW':
                return 'MEDIUM'
            elif base_severity == 'MEDIUM':
                return 'HIGH'
            else:
                return 'CRITICAL'
        elif low_count > high_count:
            # De-escalate severity
            if base_severity == 'HIGH':
                return 'MEDIUM'
            elif base_severity == 'MEDIUM':
                return 'LOW'
        
        return base_severity
    
    def _extract_indicators(self, sentence: str, category: str) -> List[str]:
        """Extract specific risk indicators from sentence"""
        indicators = []
        config = self.risk_categories[category]
        
        sentence_lower = sentence.lower()
        for keyword in config['keywords']:
            if keyword in sentence_lower:
                indicators.append(keyword)
        
        return indicators
    
    def _calculate_confidence(self, sentence: str, category: str,
                            indicators: List[str]) -> float:
        """Calculate confidence in risk identification"""
        # Base confidence on number of indicators
        confidence = min(0.5 + (len(indicators) * 0.1), 0.95)
        
        # Boost if sentence explicitly mentions risk
        if 'risk' in sentence.lower():
            confidence = min(confidence + 0.1, 0.95)
        
        return confidence
    
    def _generate_description(self, sentence: str) -> str:
        """Generate risk description"""
        # Truncate to first 100 characters
        desc = sentence[:100]
        if len(sentence) > 100:
            desc += "..."
        return desc
    
    def _deduplicate_risks(self, risks: List[Risk]) -> List[Risk]:
        """Remove duplicate or very similar risks"""
        if not risks:
            return []
        
        unique_risks = []
        seen_descriptions = set()
        
        for risk in risks:
            # Normalize description
            desc_norm = risk.description.lower()[:50]
            
            if desc_norm not in seen_descriptions:
                unique_risks.append(risk)
                seen_descriptions.add(desc_norm)
        
        return unique_risks
    
    def calculate_risk_score(self, risks: List[Risk]) -> Dict:
        """
        Calculate overall risk score
        
        Returns:
            Risk score and breakdown
        """
        if not risks:
            return {
                'total_score': 0,
                'risk_level': 'LOW',
                'risk_count': 0
            }
        
        # Severity weights
        severity_weights = {
            'CRITICAL': 10,
            'HIGH': 7,
            'MEDIUM': 4,
            'LOW': 2
        }
        
        # Calculate weighted score
        total_score = sum(
            severity_weights.get(risk.severity, 0) * risk.confidence
            for risk in risks
        )
        
        # Normalize to 0-100
        max_possible = len(risks) * 10
        normalized_score = (total_score / max_possible * 100) if max_possible > 0 else 0
        
        # Determine overall risk level
        if normalized_score >= 70:
            risk_level = 'HIGH'
        elif normalized_score >= 40:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Category breakdown
        category_counts = defaultdict(int)
        for risk in risks:
            category_counts[risk.category] += 1
        
        return {
            'total_score': normalized_score,
            'risk_level': risk_level,
            'risk_count': len(risks),
            'by_severity': {
                'CRITICAL': sum(1 for r in risks if r.severity == 'CRITICAL'),
                'HIGH': sum(1 for r in risks if r.severity == 'HIGH'),
                'MEDIUM': sum(1 for r in risks if r.severity == 'MEDIUM'),
                'LOW': sum(1 for r in risks if r.severity == 'LOW')
            },
            'by_category': dict(category_counts),
            'top_risk': risks[0].category if risks else None
        }
    
    def generate_risk_report(self, text: str) -> str:
        """Generate comprehensive risk report"""
        risks = self.analyze_risks(text)
        score = self.calculate_risk_score(risks)
        
        report = []
        
        report.append(f"\n{'='*60}")
        report.append(f"RISK ANALYSIS REPORT")
        report.append(f"{'='*60}")
        
        # Overall assessment
        report.append(f"\nOverall Risk Level: {score['risk_level']}")
        report.append(f"Risk Score: {score['total_score']:.1f}/100")
        report.append(f"Total Risks Identified: {score['risk_count']}")
        
        # By severity
        report.append(f"\nRisks by Severity:")
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = score['by_severity'].get(severity, 0)
            if count > 0:
                report.append(f"  {severity}: {count}")
        
        # By category
        report.append(f"\nRisks by Category:")
        for category, count in sorted(score['by_category'].items(), 
                                     key=lambda x: x[1], reverse=True):
            report.append(f"  {category.replace('_', ' ')}: {count}")
        
        # Detailed risks
        report.append(f"\nDetailed Risk Analysis:")
        for i, risk in enumerate(risks[:10], 1):  # Show top 10
            report.append(f"\n{i}. {risk}")
            report.append(f"   Confidence: {risk.confidence:.0%}")
            report.append(f"   Indicators: {', '.join(risk.indicators)}")
        
        report.append(f"\n{'='*60}")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    analyzer = RiskAnalyzer()
    
    # Sample text with risks
    sample_text = """
    The company faces significant market volatility and competitive pressure
    in its core markets. Supply chain disruptions continue to pose operational
    challenges. We are monitoring potential regulatory changes that could
    materially impact our business. Additionally, cybersecurity threats remain
    a concern. However, we have implemented risk mitigation strategies and
    maintain adequate liquidity to manage these challenges.
    """
    
    print("\nAnalyzing risks...")
    report = analyzer.generate_risk_report(sample_text)
    print(report)
    
    print("\n✓ Risk Analyzer Module Ready!")