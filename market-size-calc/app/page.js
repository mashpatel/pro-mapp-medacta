"use client";

import React, { useState } from 'react';
import { ChevronRight, RotateCcw } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';

const MarketSizeCalculator = () => {
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [result, setResult] = useState(null);

  const industryFactors = {
    SaaS: { growthMultiplier: 1.2, competitionFactor: 0.9 },
    Healthcare: { growthMultiplier: 1.1, competitionFactor: 0.85 },
    Fintech: { growthMultiplier: 1.15, competitionFactor: 0.95 },
    Ecommerce: { growthMultiplier: 1.1, competitionFactor: 0.8 },
    Manufacturing: { growthMultiplier: 1.05, competitionFactor: 0.9 },
    Other: { growthMultiplier: 1, competitionFactor: 1 },
  };

  const questions = [
    {
      id: 'industry',
      text: 'What industry does your product serve?',
      type: 'select',
      options: Object.keys(industryFactors).map(industry => ({ value: industry, label: industry })),
    },
    {
      id: 'totalCompanies',
      text: 'What is the total number of companies in your target market?',
      type: 'number_range',
      placeholder: 'Enter number of companies',
    },
    {
      id: 'addressablePercentage',
      text: 'What percentage of these companies could potentially use your product?',
      type: 'number',
      placeholder: 'Enter percentage',
    },
    {
      id: 'productFit',
      text: 'How well does your product fit the market needs?',
      type: 'select',
      options: [
        { value: 0.8, label: 'Excellent fit' },
        { value: 0.6, label: 'Good fit' },
        { value: 0.4, label: 'Average fit' },
        { value: 0.2, label: 'Poor fit' },
      ],
    },
    {
      id: 'competitionLevel',
      text: 'How would you describe the level of competition in your market?',
      type: 'select',
      options: [
        { value: 0.9, label: 'Low competition' },
        { value: 0.7, label: 'Moderate competition' },
        { value: 0.5, label: 'High competition' },
        { value: 0.3, label: 'Saturated market' },
      ],
    },
    {
      id: 'growthRate',
      text: 'What is the projected annual growth rate of your target market?',
      type: 'number',
      placeholder: 'Enter percentage',
    },
    {
      id: 'pricingStrategy',
      text: 'What is your average annual revenue per customer?',
      type: 'number',
      placeholder: 'Enter amount in $',
    },
    // Refined SAM questions
    {
      id: 'geographicReach',
      text: 'What percentage of the total market is within your geographic reach?',
      type: 'number',
      placeholder: 'Enter percentage',
    },
    {
      id: 'productLimitations',
      text: 'Considering your product\'s current capabilities, what percentage of the addressable market can it serve?',
      type: 'number',
      placeholder: 'Enter percentage',
    },
    {
      id: 'regulatoryConstraints',
      text: 'Are there regulatory constraints limiting your market? If so, what percentage of the market remains accessible?',
      type: 'number',
      placeholder: 'Enter percentage (100 if no constraints)',
    },
    // Existing SOM questions
    {
      id: 'expectedMarketShare',
      text: 'What market share do you realistically expect to capture in the next 3-5 years?',
      type: 'number',
      placeholder: 'Enter percentage',
    },
    {
      id: 'salesMarketingCapacity',
      text: 'What percentage of the SAM can your current sales and marketing capacity realistically reach?',
      type: 'number',
      placeholder: 'Enter percentage',
    },
  ];

  const handleInputChange = (value, subfield) => {
    if (subfield) {
      setAnswers(prev => ({
        ...prev,
        [questions[step].id]: { ...prev[questions[step].id], [subfield]: value }
      }));
    } else {
      setAnswers(prev => ({ ...prev, [questions[step].id]: value }));
    }
  };

  const handleNextStep = () => {
    if (step < questions.length - 1) {
      setStep(step + 1);
    } else {
      calculateMarkets();
    }
  };

  const calculateMarkets = () => {
    const {
      industry,
      totalCompanies,
      addressablePercentage,
      productFit,
      competitionLevel,
      growthRate,
      pricingStrategy,
      geographicReach,
      productLimitations,
      regulatoryConstraints,
      expectedMarketShare,
      salesMarketingCapacity,
    } = answers;

    const { growthMultiplier, competitionFactor } = industryFactors[industry];

    const calculateSingleTAM = (companies) => {
      const addressableCompanies = companies * (addressablePercentage / 100);
      const adjustedCompanies = addressableCompanies * productFit * (competitionLevel * competitionFactor);
      return adjustedCompanies * pricingStrategy * (1 + (growthRate / 100) * growthMultiplier);
    };

    const calculateSAM = (tam) => {
      return tam * (geographicReach / 100) * (productLimitations / 100) * (regulatoryConstraints / 100);
    };

    const calculateSOM = (sam) => {
      return sam * Math.min(expectedMarketShare, salesMarketingCapacity) / 100;
    };

    const lowTAM = calculateSingleTAM(totalCompanies.low);
    const midTAM = calculateSingleTAM(totalCompanies.mid);
    const highTAM = calculateSingleTAM(totalCompanies.high);

    const lowSAM = calculateSAM(lowTAM);
    const midSAM = calculateSAM(midTAM);
    const highSAM = calculateSAM(highTAM);

    const lowSOM = calculateSOM(lowSAM);
    const midSOM = calculateSOM(midSAM);
    const highSOM = calculateSOM(highSAM);

    setResult({
      tam: { low: Math.round(lowTAM), mid: Math.round(midTAM), high: Math.round(highTAM) },
      sam: { low: Math.round(lowSAM), mid: Math.round(midSAM), high: Math.round(highSAM) },
      som: { low: Math.round(lowSOM), mid: Math.round(midSOM), high: Math.round(highSOM) },
    });
  };

  const resetCalculator = () => {
    setStep(0);
    setAnswers({});
    setResult(null);
  };

  const renderQuestion = () => {
    const question = questions[step];
    switch (question.type) {
      case 'number':
        return (
          <Input
            type="number"
            placeholder={question.placeholder}
            value={answers[question.id] || ''}
            onChange={(e) => handleInputChange(e.target.value)}
          />
        );
      case 'number_range':
        return (
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Label htmlFor="low" className="w-20">Low estimate:</Label>
              <Input
                id="low"
                type="number"
                placeholder={question.placeholder}
                value={answers[question.id]?.low || ''}
                onChange={(e) => handleInputChange(e.target.value, 'low')}
              />
            </div>
            <div className="flex items-center space-x-2">
              <Label htmlFor="mid" className="w-20">Mid estimate:</Label>
              <Input
                id="mid"
                type="number"
                placeholder={question.placeholder}
                value={answers[question.id]?.mid || ''}
                onChange={(e) => handleInputChange(e.target.value, 'mid')}
              />
            </div>
            <div className="flex items-center space-x-2">
              <Label htmlFor="high" className="w-20">High estimate:</Label>
              <Input
                id="high"
                type="number"
                placeholder={question.placeholder}
                value={answers[question.id]?.high || ''}
                onChange={(e) => handleInputChange(e.target.value, 'high')}
              />
            </div>
          </div>
        );
      case 'select':
        return (
          <Select 
            value={answers[question.id] ? answers[question.id].toString() : undefined} 
            onValueChange={(value) => handleInputChange(question.id === 'industry' ? value : parseFloat(value))}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select an option" />
            </SelectTrigger>
            <SelectContent>
              {question.options.map((option) => (
                <SelectItem key={option.value} value={option.value.toString()}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      default:
        return null;
    }
  };

  const isAnswerValid = () => {
    const currentAnswer = answers[questions[step].id];
    if (questions[step].type === 'number_range') {
      return currentAnswer && currentAnswer.low && currentAnswer.mid && currentAnswer.high &&
             !isNaN(currentAnswer.low) && !isNaN(currentAnswer.mid) && !isNaN(currentAnswer.high);
    }
    return currentAnswer !== undefined && currentAnswer !== '' && (questions[step].type === 'select' || !isNaN(currentAnswer));
  };

  const renderResults = () => (
    <div className="space-y-4">
      {['TAM', 'SAM', 'SOM'].map((metric) => (
        <div key={metric} className="text-center">
          <h3 className="text-lg font-semibold mb-2">Estimated {metric}</h3>
          <p className="text-xl font-bold text-blue-600">
            ${result[metric.toLowerCase()].low.toLocaleString()} - ${result[metric.toLowerCase()].high.toLocaleString()}
          </p>
          <p className="text-sm text-gray-600 mt-2">
            Mid estimate: ${result[metric.toLowerCase()].mid.toLocaleString()}
          </p>
        </div>
      ))}
      <Alert className="mt-4">
        <AlertTitle>Note</AlertTitle>
        <AlertDescription>
          These ranges are based on your inputs and various factors. Actual market sizes may vary. Consider consulting with market research experts for more accurate figures.
        </AlertDescription>
      </Alert>
    </div>
  );

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Market Size Calculator</CardTitle>
        <CardDescription>Estimate your TAM, SAM, and SOM</CardDescription>
      </CardHeader>
      <CardContent>
        {result === null ? (
          <>
            <p className="mb-4">{questions[step].text}</p>
            {renderQuestion()}
          </>
        ) : renderResults()}
      </CardContent>
      <CardFooter className="flex justify-between">
        {result === null ? (
          <Button onClick={handleNextStep} disabled={!isAnswerValid()}>
            {step < questions.length - 1 ? 'Next' : 'Calculate Markets'} <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        ) : (
          <Button onClick={resetCalculator}>
            Start Over <RotateCcw className="ml-2 h-4 w-4" />
          </Button>
        )}
      </CardFooter>
    </Card>
  );
};

export default MarketSizeCalculator;