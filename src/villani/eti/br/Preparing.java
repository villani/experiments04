package villani.eti.br;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.TreeMap;
import java.util.Vector;

public class Preparing {
	
	public static void run(String id, LogBuilder log, TreeMap<String, String> entradas){
		
		log.write(" - Recebendo os parametros de entrada.");
		String caminho = entradas.get("caminho");
		int ns = Integer.parseInt(entradas.get("ns"));
		int ni = Integer.parseInt(entradas.get("ni"));
		
		log.write(" - Obtendo o conjunto principal de imagens.");
		File pasta = new File(caminho);
		File[] listaDeImagens = pasta.listFiles();
		Vector<File> cp = new Vector<File>(listaDeImagens.length);

		log.write(" - Removendo imagens do formato TIFF do conjunto.");
		for(File imagem: listaDeImagens){
			String ext = imagem.getName().split("\\.")[1];
			if(!ext.equals("tif")) cp.add(imagem);
		}
		
		log.write(" - Constituido conjunto principal com " + cp.size() + " imagens.");
		
		log.write(" - Construindo " + ns + " subconjuntos de " + ni + " imagens");
		for(int i = 0; i < ns; i++){
			
			File[] subconjunto = new File[ni];
			
			for(int j = 0; j < ni; j++){
				subconjunto[j] = cp.remove((int)(cp.size()*Math.random()));
				cp.trimToSize();
			}
			
			File subinfile = new File(id+"-Sub" + i + ".lst");
			FileWriter escritor = null;
			try {
				escritor = new FileWriter(subinfile);
				for(File imagem: subconjunto) escritor.write(imagem.getAbsolutePath() + "\n");
				escritor.close();
			} catch (IOException e) {
				log.write("Falhar ao criar arquivo " + subinfile + ": " + e.getMessage());
				System.exit(0);
			}
			
			 
		}
		
		
	}

}
