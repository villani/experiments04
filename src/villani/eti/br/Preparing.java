package villani.eti.br;

import java.io.File;
import java.util.TreeMap;
import java.util.Vector;

public class Preparing {
	
	public static void run(String id, LogBuilder log, TreeMap<String, String> entradas){
		
		log.write(" - Recebendo os parametros de entrada.");
		String caminho = entradas.get("caminho");
//		int ns = Integer.parseInt(entradas.get("ns"));
//		int ni = Integer.parseInt(entradas.get("ni"));
		
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
		
		
		
		
	}

}
