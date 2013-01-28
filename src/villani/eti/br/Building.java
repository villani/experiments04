package villani.eti.br;

import ij.ImagePlus;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.Vector;

import javax.imageio.ImageIO;

import mpi.cbg.fly.Feature;
import mpi.cbg.fly.Filter;
import mpi.cbg.fly.FloatArray2D;
import mpi.cbg.fly.FloatArray2DSIFT;
import mpi.cbg.fly.ImageArrayConverter;
import mulan.data.InvalidDataFormatException;
import net.semanticmetadata.lire.imageanalysis.mpeg7.EdgeHistogramImplementation;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import de.florianbrucker.ml.lbp.LBPModel;
import de.florianbrucker.ml.lbp.LBPParameters;

public class Building {

	public static void run(String id, LogBuilder log, TreeMap<String, String> entradas) {

		log.write(" - Recebendo os parametros de entrada.");
		boolean hasEHD = Boolean.parseBoolean(entradas.get("ehd"));
		boolean hasLBP = Boolean.parseBoolean(entradas.get("lbp"));
		int bins = Integer.parseInt(entradas.get("bins"));
		int vizinhos = Integer.parseInt(entradas.get("vizinhos"));
		int raio = Integer.parseInt(entradas.get("raio"));
		boolean hasSIFT = Boolean.parseBoolean(entradas.get("sift"));
		int histoSize = Integer.parseInt(entradas.get("histoSize"));
		boolean hasGabor = Boolean.parseBoolean(entradas.get("gabor"));
		boolean hasZernike = Boolean.parseBoolean(entradas.get("zernike"));
		String irma = entradas.get("irma");
		String rotulos = entradas.get("rotulos");
		int ns = Integer.parseInt(entradas.get("ns"));
		int ni = Integer.parseInt(entradas.get("ni"));

		log.write(" - Obtendo o conjunto de rotulos IRMA");
		XmlIrmaCodeBuilder xicb;
		try {
			log.write(" - Criando arquivo xml com a estrutura de códigos IRMA");
			xicb = new XmlIrmaCodeBuilder(irma, id);
			if (xicb.hasXml()) log.write(" - Arquivo xml com a estrutura de código IRMA criado com exito");
		} catch (IOException e) {
			log.write(" - Falha ao obter relacao nome da imagem/ codigo IRMA: " + e.getMessage());
			System.exit(0);
		}
		
		log.write(" - Obtendo a relacao nome da imagem/código IRMA do arquivo: " + rotulos);
		File relacaoImagemCodigo = new File(rotulos);
		TreeMap<String, String> relacao = new TreeMap<String, String>();
		Scanner leitor = null; 
		try {
			leitor = new Scanner(relacaoImagemCodigo);
		} catch (FileNotFoundException fnfe) {
			log.write(" - Falha ao ler o arquivo ao obter relacao" + rotulos + ": " + fnfe.getMessage());
			System.exit(0);
		}
		while (leitor.hasNextLine()) {
			String[] campos = leitor.nextLine().split(";");
			relacao.put(campos[0], campos[1]);
		}
		leitor.close();

		log.write(" - Criando objeto que converte o codigo IRMA para binario e que tambem necessita do xml criado anteriormente");
		IrmaCode conversor = null;
		try {
			conversor = new IrmaCode(id);
		} catch (FileNotFoundException fnfe) {
			log.write("Falha ao ler o arquivo xml " + id + ".xml: " + fnfe.getMessage());
			System.exit(0);
		}

		if (hasEHD) {
			for (int i = 0; i < ns; i++) {
				log.write(" - Construindo a base ARFF EHD para o subconjunto " + i);
				
				String dataset = id + "-Ehd-Sub" + i;
				RelationBuilder instanciasEHD = null;
				try{
					instanciasEHD = new RelationBuilder(dataset, id);
					for (int j = 0; j < 80; j++) instanciasEHD.defineAttribute("ehd" + j, "numeric");
					log.write(" - Salvando a lista de atributos e incluindo a lista de rótulos a partir do xml");
					instanciasEHD.saveAttributes();
				} catch(IOException ioe){
					log.write(" - Falha ao criar ou salvar a lista de atributos e rótulos: " + ioe.getMessage());
					System.exit(0);
				} catch(Exception e){
					log.write(" - Falha ao define um atributo: " + e.getMessage());
					System.exit(0);
				}

				log.write(" - Obtendo caracteristicas EHD para cada imagem");
				ArrayList<File> imagens = new ArrayList<File>(ni);
				File subconjunto = new File(id + "-Sub" + i + ".lst");
				leitor = null;
				try {
					leitor = new Scanner(subconjunto);
				} catch (FileNotFoundException fnfe) {
					log.write(" - Falha ao ler o arquivo " + subconjunto.getName() + " ao obter EHD: " + fnfe.getMessage());
					System.exit(0);
				}
				
				while (leitor.hasNextLine()) imagens.add(new File(leitor.nextLine()));
				for (File imagem : imagens) {
					EdgeHistogramImplementation extrator = null;
					try {
						extrator = new EdgeHistogramImplementation(ImageIO.read(imagem));
					} catch (IOException ioe) {
						log.write(" - Falha ao ler imagem " + imagem.getName() + " ao extrair características: " + ioe.getMessage());
						System.exit(0);
					}
					int[] ehd = extrator.setEdgeHistogram();
					String amostra = "";
					for (int e : ehd) amostra += e + ",";
					String nomeImg = imagem.getName().split("\\.")[0];
					amostra += conversor.toBinary(relacao.get(nomeImg));
					try {
						instanciasEHD.insertData(amostra);
					} catch (Exception ex) {
						log.write(" - Falha ao inserir amostra" + amostra + ": " + ex.getMessage());
						System.exit(0);
					}
				}

				try {
					instanciasEHD.saveRelation();
				} catch (InvalidDataFormatException idfe){
					log.write(" - Falha no formato ao salvar relação: " + idfe.getMessage());
				} catch (IOException ioe) {
					log.write(" - Falha ao salvar relação: " + ioe.getMessage());
				}
				log.write(" - Novo conjunto de amostras salvo em: " + dataset + ".arff");
			}
		}

		if (hasLBP){
			for (int i = 0; i < ns; i++) {
				log.write(" - Construindo a base ARFF LBP para o subconjunto " + i);
				
				String dataset = id + "-Lbp-Sub" + i;
				RelationBuilder instanciasLBP = null;
				try{
					instanciasLBP = new RelationBuilder(dataset, id);
					for (int j = 0; j < bins; j++) instanciasLBP.defineAttribute("lbp" + j, "numeric");
					log.write("- Salvando a lista de atributos e incluindo a lista de rotulos a partir do xml");
					instanciasLBP.saveAttributes();
				} catch(IOException ioe){
					log.write(" - Falha ao criar ou salvar a lista de atributos e rótulos: " + ioe.getMessage());
					System.exit(0);
				} catch(Exception e){
					log.write(" - Falha ao define um atributo: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Obtendo caracteristicas LBP para cada imagem");
				ArrayList<File> imagens = new ArrayList<File>(ni);
				File subconjunto = new File(id + "-Sub" + i + ".lst");
				leitor = null;
				try {
					leitor = new Scanner(subconjunto);
				} catch (FileNotFoundException fnfe) {
					log.write(" - Falha ao ler o arquivo " + subconjunto.getName() + " ao obter LBP: " + fnfe.getMessage());
					System.exit(0);
				}
				
				while (leitor.hasNextLine()) imagens.add(new File(leitor.nextLine()));
				for (File imagem : imagens) {
					LBPParameters p = new LBPParameters(vizinhos, raio, bins);
					LBPModel extrator = null;
					try {
						extrator = new LBPModel(p, imagem);
					} catch (IOException ioe) {
						log.write(" - Falha ao ler imagem " + imagem.getName() + " ao extrair características: " + ioe.getMessage());
						System.exit(0);
					}
					float[] histLBP = extrator.subModels[0].patternHist;
					String amostra = "";
					for (float lbp : histLBP) amostra += lbp + ",";
					String nomeImg = imagem.getName().split("\\.")[0];
					amostra += conversor.toBinary(relacao.get(nomeImg));
					try {
						instanciasLBP.insertData(amostra);
					} catch (Exception ex) {
						log.write(" - Falha ao inserir amostra" + amostra + ": " + ex.getMessage());
						System.exit(0);
					}
				}

				try {
					instanciasLBP.saveRelation();
				} catch (InvalidDataFormatException idfe){
					log.write(" - Falha no formato ao salvar relação: " + idfe.getMessage());
				} catch (IOException ioe) {
					log.write(" - Falha ao salvar relação: " + ioe.getMessage());
				}
				log.write(" - Novo conjunto de amostras salvo em: " + dataset + ".arff");
			}
		}
		
		if (hasSIFT){
			for (int i = 0; i < ns; i++) {
				log.write(" - Construindo a base ARFF SIFT para o subconjunto " + i);
				
				String dataset = id + "-Sift-Sub" + i;
				RelationBuilder instanciasSIFT = null;
				try{
					instanciasSIFT = new RelationBuilder(dataset, id);
					for(int j = 0; j < histoSize; j++) instanciasSIFT.defineAttribute("histSIFT" + j, "numeric");
					log.write(" - Salvando a lista de atributos e incluindo a lista de rotulos a partir do xml");
					instanciasSIFT.saveAttributes();
				} catch(IOException ioe){
					log.write(" - Falha ao criar ou salvar a lista de atributos e rótulos: " + ioe.getMessage());
					System.exit(0);
				} catch(Exception e){
					log.write(" - Falha ao define um atributo: " + e.getMessage());
					System.exit(0);
				}
				
				log.write(" - Obtendo caracteristicas SIFT para as imagens");
				ArrayList<File> imagens = new ArrayList<File>(ni);
				File subconjunto = new File(id + "-Sub" + i + ".lst");
				leitor = null;
				try {
					leitor = new Scanner(subconjunto);
				} catch (FileNotFoundException fnfe) {
					log.write(" - Falha ao ler o arquivo " + subconjunto.getName() + " ao obter SIFT: " + fnfe.getMessage());
					System.exit(0);
				}
				
				while (leitor.hasNextLine()) imagens.add(new File(leitor.nextLine()));
				
				// Início - Identificação dos pontos-chave para cálculo do histograma SIFT
				ArrayList<String> idPontos = new ArrayList<String>();
				ArrayList<Attribute> listaDeAtributos = new ArrayList<Attribute>();
				for(int j = 0; j < 128; j++) listaDeAtributos.add(new Attribute("feat" + j));
				Instances instancias = new Instances("sift",listaDeAtributos,10);
				for(File imagem : imagens){
					ImagePlus ip = new ImagePlus(imagem.getAbsolutePath());
					FloatArray2DSIFT sift = new FloatArray2DSIFT(4,8);
					FloatArray2D fa = ImageArrayConverter.ImageToFloatArray2D(ip.getProcessor().convertToFloat());
					Filter.enhance(fa, 1.0f);
					float initial_sigma = 1.6f;
					fa = Filter.computeGaussianFastMirror(fa, (float)Math.sqrt(initial_sigma * initial_sigma - 0.25));
					sift.init(fa, 3, initial_sigma, 64, 1024);
					Vector<Feature> pontosChave = sift.run(1024);
					for(Feature ponto : pontosChave){
						idPontos.add(imagem.getName());
						Instance instancia = new DenseInstance(128);
//						instancia.setDataset(instancias);
						for(int j = 0; j < ponto.descriptor.length; j++) instancia.setValue(j, ponto.descriptor[j]);
						instancias.add(instancia);
					}
				}
				SimpleKMeans km = new SimpleKMeans();
				int[] atribuicoes = null;
				try{
					km.setNumClusters(histoSize);
					km.setOptions(new String[]{"-O","-fast"});
					km.buildClusterer(instancias);
					atribuicoes = km.getAssignments();
				} catch(Exception e){
					log.write(" - Falha no agrupamento dos pontos-chave para o Histograma SIFT: " + e.getMessage());
					System.exit(0);
				}
				TreeMap<String,int[]> histoSIFT = new TreeMap<String, int[]>();
				for(int j = 0; j < atribuicoes.length; j++){
					String img = idPontos.get(j); 
					if(! histoSIFT.containsKey(img)) histoSIFT.put(img, new int[histoSize]);
					histoSIFT.get(img)[atribuicoes[i]]++;
				}
				// Fim - Identificação dos pontos-chave para cálculo do histograma SIFT
				
				for (File imagem : imagens) {					
	
					String amostra = "";
					int[] histograma = histoSIFT.get(imagem.getName());
					for(int h : histograma) amostra += h + ",";					
					String nomeImg = imagem.getName().split("\\.")[0];					
					amostra += conversor.toBinary(relacao.get(nomeImg));
					try {
						instanciasSIFT.insertData(amostra);
					} catch (Exception ex) {
						log.write(" - Falha ao inserir amostra" + amostra + ": " + ex.getMessage());
						System.exit(0);
					}
					
				}

				try {
					instanciasSIFT.saveRelation();
				} catch (InvalidDataFormatException idfe){
					log.write(" - Falha no formato ao salvar relação: " + idfe.getMessage());
				} catch (IOException ioe) {
					log.write(" - Falha ao salvar relação: " + ioe.getMessage());
				}
				log.write(" - Novo conjunto de amostras salvo em: " + dataset + ".arff");
			}
		}
		
		if (hasGabor){
			
		}
		
		if (hasZernike){
			
		}

	}

}
