package villani.eti.br;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.TreeMap;

import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.ClassifierChain;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;


public class Evaluating {
	
	public static void run(String id, LogBuilder log, TreeMap<String, String> entradas){
		
		log.write(" - Recebendo os parametros de entrada.");
		int ns = Integer.parseInt(entradas.get("ns"));
		boolean hasMLkNN = Boolean.parseBoolean(entradas.get("mlknn"));
		boolean hasBRkNN = Boolean.parseBoolean(entradas.get("brknn"));
		boolean hasChain = Boolean.parseBoolean(entradas.get("chain"));
		boolean hasClus = Boolean.parseBoolean(entradas.get("clus"));
		boolean hasEHD = Boolean.parseBoolean(entradas.get("ehd"));
		boolean hasLBP = Boolean.parseBoolean(entradas.get("lbp"));
		boolean hasSIFT = Boolean.parseBoolean(entradas.get("sift"));
		boolean hasGabor = Boolean.parseBoolean(entradas.get("gabor"));
		boolean hasZernike = Boolean.parseBoolean(entradas.get("zernike"));
		
		
		if(hasMLkNN){
			
			if(hasEHD){
				
				log.write(" - Instanciando subconjunto 0 para treinamento EHD");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Ehd-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste EHD");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Ehd-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Ehd-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
			if(hasLBP){
				
				log.write(" - Instanciando subconjunto 0 para treinamento LBP");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Lbp-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste LBP");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Lbp-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Lbp-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}	
				
			}
			
			if(hasSIFT){
				
				log.write(" - Instanciando subconjunto 0 para treinamento SIFT");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Sift-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste SIFT");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Sift-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Sift-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
				
			}
			
			if(hasGabor){
				
				log.write(" - Instanciando subconjunto 0 para treinamento Gabor");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Gabor-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste Gabor");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Gabor-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Gabor-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
			if(hasZernike){
				
				log.write(" - Instanciando subconjunto 0 para treinamento Zernike");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Zernike-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste Zernike");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Zernike-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Zernike-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
		}
		
		if(hasBRkNN){
			
			if(hasEHD){
				
				log.write(" - Instanciando subconjunto 0 para treinamento EHD");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Ehd-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador BRkNN");
				BRkNN classifier = new BRkNN(10);
				
				log.write(" - Construindo modelo do BRkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do BRkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste EHD");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Ehd-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador BRkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-BRkNN-Ehd-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do BRkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
			if(hasLBP){
				
				log.write(" - Instanciando subconjunto 0 para treinamento LBP");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Lbp-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador BRkNN");
				BRkNN classifier = new BRkNN(10);
				
				log.write(" - Construindo modelo do BRkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do BRkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste LBP");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Lbp-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador BRkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-BRkNN-Lbp-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do BRkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}	
				
			}
			
			if(hasSIFT){
				
				log.write(" - Instanciando subconjunto 0 para treinamento SIFT");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Sift-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador BRkNN");
				BRkNN classifier = new BRkNN(10);
				
				log.write(" - Construindo modelo do BRkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do BRkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste SIFT");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Sift-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador BRkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-BRkNN-Sift-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do BRkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
				
			}
			
			if(hasGabor){
				
				log.write(" - Instanciando subconjunto 0 para treinamento Gabor");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Gabor-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador BRkNN");
				BRkNN classifier = new BRkNN(10);
				
				log.write(" - Construindo modelo do BRkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do BRkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste Gabor");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Gabor-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador BRkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-BRkNN-Gabor-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do BRkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
			if(hasZernike){
				
				log.write(" - Instanciando subconjunto 0 para treinamento Zernike");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Zernike-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador BRkNN");
				BRkNN classifier = new BRkNN(10);
				
				log.write(" - Construindo modelo do BRkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do BRkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste Zernike");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Zernike-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador BRkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-BRkNN-Zernike-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do BRkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
		}
		
		if(hasChain){
			
			if(hasEHD){
				
				log.write(" - Instanciando subconjunto 0 para treinamento EHD");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Ehd-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador Chain");
				ClassifierChain classifier = new ClassifierChain();
				
				log.write(" - Construindo modelo do Chain a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do Chain: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste EHD");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Ehd-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador Chain");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-Chain-Ehd-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do Chain\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
			if(hasLBP){
				
				log.write(" - Instanciando subconjunto 0 para treinamento LBP");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Lbp-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador Chain");
				ClassifierChain classifier = new ClassifierChain();
				
				log.write(" - Construindo modelo do Chain a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do Chain: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste LBP");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Lbp-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador Chain");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-Chain-Lbp-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do Chain\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}	
				
			}
			
			if(hasSIFT){
				
				log.write(" - Instanciando subconjunto 0 para treinamento SIFT");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Sift-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador Chain");
				ClassifierChain classifier = new ClassifierChain();
				
				log.write(" - Construindo modelo do Chain a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do Chain: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste SIFT");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Sift-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador Chain");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-Chain-Sift-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do Chain\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
				
			}
			
			if(hasGabor){
				
				log.write(" - Instanciando subconjunto 0 para treinamento Gabor");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Gabor-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador Chain");
				ClassifierChain classifier = new ClassifierChain();
				
				log.write(" - Construindo modelo do Chain a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do Chain: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste Gabor");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Gabor-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador Chain");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-Chain-Gabor-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do Chain\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
			if(hasZernike){
				
				log.write(" - Instanciando subconjunto 0 para treinamento Zernike");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Zernike-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador Chain");
				ClassifierChain classifier = new ClassifierChain();
				
				log.write(" - Construindo modelo do Chain a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do Chain: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste Zernike");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Zernike-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador Chain");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-Chain-Zernike-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do Chain\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
		}
		
		if(hasClus){
			
			if(hasEHD){
				
				log.write(" - Instanciando subconjunto 0 para treinamento EHD");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Ehd-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste EHD");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Ehd-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Ehd-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
			if(hasLBP){
				
				log.write(" - Instanciando subconjunto 0 para treinamento LBP");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Lbp-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste LBP");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Lbp-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Lbp-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}	
				
			}
			
			if(hasSIFT){
				
				log.write(" - Instanciando subconjunto 0 para treinamento SIFT");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Sift-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste SIFT");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Sift-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Sift-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
				
			}
			
			if(hasGabor){
				
				log.write(" - Instanciando subconjunto 0 para treinamento Gabor");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Gabor-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste Gabor");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Gabor-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Gabor-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
			if(hasZernike){
				
				log.write(" - Instanciando subconjunto 0 para treinamento Zernike");
				MultiLabelInstances trainSet = null;
				try {
					 trainSet = new MultiLabelInstances(id+"-Zernike-Sub0.arff", id + ".xml");
				} catch (InvalidDataFormatException idfe) {
					log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando classificador MLkNN");
				MLkNN classifier = new MLkNN();
				
				log.write(" - Construindo modelo do MLkNN a partir do conjunto de treinamento");
				try {
					classifier.build(trainSet);
				} catch (Exception e) {
					log.write(" - Falha ao construir modelo do MLkNN: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Instanciando avaliador");
				Evaluator avaliador = new Evaluator();
				
				log.write(" - Instanciando lista de medidas");
				ArrayList<Measure> medidas = new ArrayList<Measure>();
				medidas.add(new HammingLoss());
                medidas.add(new SubsetAccuracy());
                medidas.add(new ExampleBasedPrecision());
                medidas.add(new ExampleBasedRecall());
                medidas.add(new ExampleBasedFMeasure());
                medidas.add(new ExampleBasedAccuracy());
                medidas.add(new ExampleBasedSpecificity());
                int numOfLabels = trainSet.getNumLabels();
                medidas.add(new MicroPrecision(numOfLabels));
                medidas.add(new MicroRecall(numOfLabels));
                medidas.add(new MicroFMeasure(numOfLabels));
                medidas.add(new AveragePrecision());
                medidas.add(new Coverage());
                medidas.add(new OneError());
                medidas.add(new IsError());
                medidas.add(new ErrorSetSize());
                medidas.add(new RankingLoss());
				
				for(int i = 1; i < ns; i++) {
					log.write(" - Instanciando subconjunto " + i + " para teste Zernike");
					MultiLabelInstances testSet = null; 
					try {
						testSet = new MultiLabelInstances(id+"-Zernike-Sub" + i + ".arff", id + ".xml");
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Falha ao carregar formato do conjunto de treinamento: " + idfe.getMessage());
						System.exit(0);
					}
					
					log.write(" - Avaliando o modelo gerado pelo classificador MLkNN");
					Evaluation avaliacao = null;
					try {
						 avaliacao = avaliador.evaluate(classifier, testSet, medidas);
					} catch (IllegalArgumentException iae){
						log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
						System.exit(0);
					} catch (Exception e) {
						log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
						System.exit(0);
					}
					
					log.write(" - Salvando resultado da avaliação");
					File resultado = new File(id+"-MLkNN-Zernike-Sub" + i + ".result");
					try{
						FileWriter escritor = new FileWriter(resultado);
						escritor.write("=> Avaliação do MLkNN\n\n");
						escritor.write(avaliacao.toString());
						escritor.close();
					} catch(IOException ioe){
						log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
						System.exit(0);
					}
					
				}
				
			}
			
		}
		
	}

}
