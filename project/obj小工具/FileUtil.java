import java.io.*;
import java.util.*;
public class FileUtil {
	public static Scanner openInFile(String filename){
		Scanner sc = null;
		try{
			sc = new Scanner (new FileInputStream(filename));
		}catch(IOException ioe){
			ioe.printStackTrace();
		}
		return sc;
	}
	public static PrintStream openOutFile(String filename){
		PrintStream ps = null;
		try{
			ps = new PrintStream (new FileOutputStream(filename));
		}catch(IOException ioe){
			ioe.printStackTrace();
		}
		return ps;
	}
}
