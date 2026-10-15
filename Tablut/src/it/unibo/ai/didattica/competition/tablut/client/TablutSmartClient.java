package it.unibo.ai.didattica.competition.tablut.client;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.net.UnknownHostException;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.google.gson.Gson;

import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * 
 * @author A. Piretti, Andrea Galassi, Luca Peratella
 *
 */
public class TablutSmartClient extends TablutClient {

	private int game;

	public TablutSmartClient(String player, String name, int gameChosen, int timeout, String ipAddress)
			throws UnknownHostException, IOException {
		super(player, name, timeout, ipAddress);
		game = gameChosen;
	}

	public TablutSmartClient(String player, String name, int timeout, String ipAddress)
			throws UnknownHostException, IOException {
		this(player, name, 4, timeout, ipAddress);
	}

	public TablutSmartClient(String player, int timeout, String ipAddress) throws UnknownHostException, IOException {
		this(player, "smart", 4, timeout, ipAddress);
	}

	public TablutSmartClient(String player) throws UnknownHostException, IOException {
		this(player, "smart", 4, 60, "localhost");
	}

	public static void main(String[] args) throws UnknownHostException, IOException, ClassNotFoundException {
		int gametype = 4;
		String role = "";
		String name = "smart";
		String ipAddress = "localhost";
		int timeout = 60;
		// TODO: change the behavior?
		if (args.length < 1) {
			System.out.println("You must specify which player you are (WHITE or BLACK)");
			System.exit(-1);
		} else {
			System.out.println(args[0]);
			role = (args[0]);
		}
		if (args.length == 2) {
			System.out.println(args[1]);
			timeout = Integer.parseInt(args[1]);
		}
		if (args.length == 3) {
			ipAddress = args[2];
		}
		System.out.println("Selected client: " + args[0]);

		TablutSmartClient client = new TablutSmartClient(role, name, gametype, timeout, ipAddress);
		client.run();
	}

	@Override
	public void run() {

		try {
			this.declareName();
		} catch (Exception e) {
			e.printStackTrace();
		}

		State state;

		Game rules = null;
		switch (this.game) {
			case 1:
				state = new StateTablut();
				rules = new GameTablut();
				break;
			case 2:
				state = new StateTablut();
				rules = new GameModernTablut();
				break;
			case 3:
				state = new StateBrandub();
				rules = new GameTablut();
				break;
			case 4:
				state = new StateTablut();
				state.setTurn(State.Turn.WHITE);
				rules = new GameAshtonTablut(99, 0, "garbage", "fake", "fake");
				System.out.println("Ashton Tablut game");
				break;
			default:
				System.out.println("Error in game selection");
				System.exit(4);
		}

		List<int[]> pawns = new ArrayList<int[]>();
		List<int[]> empty = new ArrayList<int[]>();

		System.out.println("You are player " + this.getPlayer().toString() + "!");

		while (true) {
			try {
				this.read();
			} catch (ClassNotFoundException | IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
				System.exit(1);
			}
			System.out.println("Current state:");
			state = this.getCurrentState();
			System.out.println(state.toString());
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
			}

			if (this.getPlayer().equals(Turn.WHITE)) {
				// Mio turno
				if (this.getCurrentState().getTurn().equals(StateTablut.Turn.WHITE)) {

					/*
					 * function negamax(nodo, profondità, α, β)
					 * if nodo è un nodo terminale or profondità = 0
					 * return valore euristico del nodo
					 * else
					 * foreach figlio
					 * α := max(α, -negamax(figlio, profondità-1, -β, -α))
					 * {ciò che segue, se verificato, costituisce la potatura alfa-beta}
					 * if α ≥ β
					 * return β
					 * return α
					 */

					boolean found = false;
					Action a = null;
					try {
						a = new Action("z0", "z0", State.Turn.WHITE);
					} catch (IOException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}
					while (!found) {
						a = findBestWhiteMove(
								state,
								rules,
								5);

						try {
							rules.checkMove(state, a);
							found = true;
						} catch (Exception e) {

						}
					}

					System.out.println("Mossa scelta: " + a.toString());
					try {
						this.write(a);
					} catch (ClassNotFoundException | IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					pawns.clear();
					empty.clear();

				}
				// Turno dell'avversario
				else if (state.getTurn().equals(StateTablut.Turn.BLACK)) {
					System.out.println("Waiting for your opponent move... ");
				}
				// ho vinto
				else if (state.getTurn().equals(StateTablut.Turn.WHITEWIN)) {
					System.out.println("YOU WIN!");
					System.exit(0);
				}
				// ho perso
				else if (state.getTurn().equals(StateTablut.Turn.BLACKWIN)) {
					System.out.println("YOU LOSE!");
					System.exit(0);
				}
				// pareggio
				else if (state.getTurn().equals(StateTablut.Turn.DRAW)) {
					System.out.println("DRAW!");
					System.exit(0);
				}

			} else {

				// Mio turno
				if (this.getCurrentState().getTurn().equals(StateTablut.Turn.BLACK)) {

					boolean found = false;
					Action a = null;
					try {
						a = new Action("z0", "z0", State.Turn.BLACK);
					} catch (IOException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}
					;
					while (!found) {
						a = findBestBlackMove(
								state,
								rules,
								5);

						System.out.println("try: " + a.toString());
						try {
							rules.checkMove(state, a);
							found = true;
						} catch (Exception e) {

						}

					}

					System.out.println("Mossa scelta: " + a.toString());
					try {
						this.write(a);
					} catch (ClassNotFoundException | IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					pawns.clear();
					empty.clear();

				}

				else if (state.getTurn().equals(StateTablut.Turn.WHITE)) {
					System.out.println("Waiting for your opponent move... ");
				} else if (state.getTurn().equals(StateTablut.Turn.WHITEWIN)) {
					System.out.println("YOU LOSE!");
					System.exit(0);
				} else if (state.getTurn().equals(StateTablut.Turn.BLACKWIN)) {
					System.out.println("YOU WIN!");
					System.exit(0);
				} else if (state.getTurn().equals(StateTablut.Turn.DRAW)) {
					System.out.println("DRAW!");
					System.exit(0);
				}

			}
		}

	}

	private Action findBestWhiteMove(State state, Game rules, int depth) {

		// Sincronizza il turno
		State searchState = deepCopyState(state);
		searchState.setTurn(State.Turn.WHITE);

		List<Action> legalMoves = generateWhiteMoves(searchState, rules);

		List<Action> kingMoves = new ArrayList<>();
		List<Action> pawnMoves = new ArrayList<>();

		Action bestMove = null;
		int bestValue = Integer.MIN_VALUE;

		for (Action move : legalMoves) {
			State.Pawn pawn = searchState.getPawn(move.getRowFrom(), move.getColumnFrom());
			if (pawn.equalsPawn(State.Pawn.KING.toString())) {
				kingMoves.add(move);
			} else {
				pawnMoves.add(move);
			}
		}

		Action bestKingMove = null;
		int bestKingValue = Integer.MIN_VALUE;

		for (Action move : kingMoves) {
			State testState = deepCopyState(searchState);
			movePawn(testState, move);
			int value = negamax(testState, rules, State.Turn.WHITE, depth - 1);

			System.out.println("  👑 RE move: " + move + " → Value: " + value);

			if (value > bestKingValue) {
				bestKingValue = value;
				bestKingMove = move;
			}
		}

		// 🔴 Valuta mosse delle PEDINE (priorità: ATTACCO)
		Action bestPawnMove = null;
		int bestPawnValue = Integer.MIN_VALUE;

		for (Action move : pawnMoves) {
			State testState = deepCopyState(searchState);
			movePawn(testState, move);
			int value = negamax(testState, rules, State.Turn.WHITE, depth - 1);

			System.out.println("  ⚪ PAWN move: " + move + " → Value: " + value);

			if (value > bestPawnValue) {
				bestPawnValue = value;
				bestPawnMove = move;
			}
		}

		// 🔴 DECIDE: RE > PEDINE (il RE deve sopravvivere)
		if (bestKingValue > bestPawnValue) {
			System.out.println("🔵 WHITE chooses KING move: " + bestKingMove + " (value: " + bestKingValue + ")");
			return bestKingMove;
		} else {
			System.out.println("🔵 WHITE chooses PAWN move: " + bestPawnMove + " (value: " + bestPawnValue + ")");
			return bestPawnMove;
		}
	}

	private Action findBestBlackMove(State state, Game rules, int depth) {

		// Sincronizza il turno
		State searchState = deepCopyState(state);
		searchState.setTurn(State.Turn.BLACK);

		List<Action> legalMoves = generateBlackMoves(searchState, rules);

		Action bestMove = null;
		int bestValue = Integer.MIN_VALUE;

		for (Action move : legalMoves) {
			State nextState = applyMoveCopy(searchState, move);

			int value = -negamax(
					nextState,
					rules,
					State.Turn.WHITE,
					depth - 1);

			if (value > bestValue) {
				bestValue = value;
				bestMove = move;
			}
		}

		return bestMove;
	}

	private List<Action> generateWhiteMoves(
			State state,
			Game rules) {

		List<Action> moves = new ArrayList<>();

		int boardSize = state.getBoard().length;

		for (int fromRow = 0; fromRow < boardSize; fromRow++) {
			for (int fromCol = 0; fromCol < boardSize; fromCol++) {

				State.Pawn pawn = state.getPawn(fromRow, fromCol);

				boolean isWhitePawn = pawn.equalsPawn(State.Pawn.WHITE.toString())
						|| pawn.equalsPawn(State.Pawn.KING.toString());

				if (!isWhitePawn) {
					continue;
				}

				String from = state.getBox(fromRow, fromCol);

				for (int toRow = 0; toRow < boardSize; toRow++) {
					for (int toCol = 0; toCol < boardSize; toCol++) {

						if (!state.getPawn(toRow, toCol)
								.equalsPawn(State.Pawn.EMPTY.toString())) {
							continue;
						}

						String to = state.getBox(toRow, toCol);

						try {
							Action action = new Action(
									from,
									to,
									State.Turn.WHITE);

							rules.checkMove(state, action);
							moves.add(action);

						} catch (Exception e) {
							// La mossa non è valida
						}
					}
				}
			}
		}

		return moves;
	}

	private List<Action> generateBlackMoves(
			State state,
			Game rules) {

		List<Action> moves = new ArrayList<>();

		int boardSize = state.getBoard().length;

		for (int fromRow = 0; fromRow < boardSize; fromRow++) {
			for (int fromCol = 0; fromCol < boardSize; fromCol++) {

				State.Pawn pawn = state.getPawn(fromRow, fromCol);

				boolean isBlackPawn = pawn.equalsPawn(State.Pawn.BLACK.toString());

				if (!isBlackPawn) {
					continue;
				}

				String from = state.getBox(fromRow, fromCol);

				for (int toRow = 0; toRow < boardSize; toRow++) {
					for (int toCol = 0; toCol < boardSize; toCol++) {

						if (!state.getPawn(toRow, toCol)
								.equalsPawn(State.Pawn.EMPTY.toString())) {
							continue;
						}

						String to = state.getBox(toRow, toCol);

						try {
							Action action = new Action(
									from,
									to,
									State.Turn.BLACK);

							rules.checkMove(state, action);
							moves.add(action);

						} catch (Exception e) {
							// La mossa non è valida
						}
					}
				}
			}
		}

		return moves;
	}

	private int negamax(
			State state,
			Game rules,
			State.Turn player,
			int depth) {

		if (depth == 0 || isTerminalState(state)) {
			return evaluate(state, player);
		}

		// Sincronizza il turno dello stato con il giocatore
		State currentState = deepCopyState(state);
		currentState.setTurn(player);

		List<Action> legalMoves;

		if (player.equals(State.Turn.WHITE)) {
			legalMoves = generateWhiteMoves(currentState, rules);
		} else {
			legalMoves = generateBlackMoves(currentState, rules);
		}

		if (legalMoves.isEmpty()) {
			return evaluate(currentState, player);
		}

		int bestValue = Integer.MIN_VALUE;

		for (Action move : legalMoves) {
			State nextState = applyMoveCopy(currentState, move);

			State.Turn opponent = player.equals(State.Turn.WHITE)
					? State.Turn.BLACK
					: State.Turn.WHITE;

			int value = -negamax(
					nextState,
					rules,
					opponent,
					depth - 1);

			if (value > bestValue) {
				bestValue = value;
			}
		}

		return bestValue;
	}

	private int evaluate(State state, State.Turn player) {

		// Controlli di stato terminale
		if (state.getTurn().equals(State.Turn.WHITEWIN)) {
			return player.equals(State.Turn.WHITE) ? 100000 : -100000;
		}

		if (state.getTurn().equals(State.Turn.BLACKWIN)) {
			return player.equals(State.Turn.BLACK) ? 100000 : -100000;
		}

		if (state.getTurn().equals(State.Turn.DRAW)) {
			return 0;
		}

		State.Pawn[][] board = state.getBoard();
		int boardSize = board.length;

		// Trova il RE
		int kingRow = -1, kingCol = -1;
		for (int row = 0; row < boardSize; row++) {
			for (int col = 0; col < boardSize; col++) {
				if (state.getPawn(row, col).equalsPawn(State.Pawn.KING.toString())) {
					kingRow = row;
					kingCol = col;
					break;
				}
			}
			if (kingRow != -1)
				break;
		}

		if (kingRow == -1) {
			return player.equals(State.Turn.BLACK) ? 100000 : -100000;
		}

		// Conta le pedine
		int whitePieces = 0;
		int blackPieces = 0;

		for (int row = 0; row < boardSize; row++) {
			for (int col = 0; col < boardSize; col++) {
				State.Pawn pawn = state.getPawn(row, col);
				if (pawn.equalsPawn(State.Pawn.WHITE.toString())) {
					whitePieces++;
				} else if (pawn.equalsPawn(State.Pawn.BLACK.toString())) {
					blackPieces++;
				}
			}
		}

		// ✅ NUOVA VALUTAZIONE CON OBIETTIVI SEPARATI
		int kingEscapeDistance = calculateKingEscapeDistance(kingRow, kingCol, boardSize);
		int kingThreatDistance = calculateKingThreatDistance(kingRow, kingCol, board, boardSize);

		int score;

		if (player.equals(State.Turn.WHITE)) {
			// 🔴 OBIETTIVO RE: fuggire (minimizzare distanza + evitare minacce)
			int kingScore = -1000 * kingEscapeDistance // MASSIMA PRIORITÀ: ogni passo verso fuga = +1000
					+ kingThreatDistance * 10; // Bonus se lontano dai neri

			// 🔴 OBIETTIVO PEDINE: attaccare e sopravvivere
			int pawnScore = (whitePieces * 50) // Pedine bianche valgono 50 punti
					- (blackPieces * 60); // Pedine nere valgono 60 punti (difesa prioritaria)

			// Combina: RE ha priorità maggiore
			score = kingScore + pawnScore;

		} else {
			// Black vuole catturare il RE e controllare le pedine bianche
			int kingScore = 1000 * kingEscapeDistance // PENALITÀ: re che scappa è male per black
					- kingThreatDistance * 10; // BONUS: re minacciato è bene per black

			int pawnScore = (blackPieces * 50)
					- (whitePieces * 60);

			score = kingScore + pawnScore;
		}

		return score;
	}

	/**
	 * Calcola quanto il RE è minacciato dai neri
	 * (bonus se circondato, penalità se libero)
	 */

	private int calculateKingThreatDistance(int kingRow, int kingCol, State.Pawn[][] board, int boardSize) {
		int minDistance = Integer.MAX_VALUE;

		for (int row = 0; row < boardSize; row++) {
			for (int col = 0; col < boardSize; col++) {
				if (board[row][col].equalsPawn(State.Pawn.BLACK.toString())) {
					int distance = Math.abs(kingRow - row) + Math.abs(kingCol - col);
					minDistance = Math.min(minDistance, distance);
				}
			}
		}

		return minDistance == Integer.MAX_VALUE ? 0 : minDistance;
	}

	/**
	 * Calcola la distanza minima del RE dagli angoli (celle di fuga)
	 */
	private int calculateKingEscapeDistance(int kingRow, int kingCol, int boardSize) {
		int[][] escapes = { { 0, 0 }, { 0, boardSize - 1 }, { boardSize - 1, 0 }, { boardSize - 1, boardSize - 1 } };
		int minDistance = Integer.MAX_VALUE;

		for (int[] escape : escapes) {
			int distance = Math.abs(kingRow - escape[0]) + Math.abs(kingCol - escape[1]);
			minDistance = Math.min(minDistance, distance);
		}

		return minDistance;
	}

	private boolean isEscapeCell(int row, int col, int boardSize) {
		return (row == 0 || row == boardSize - 1) && (col == 0 || col == boardSize - 1);
	}

	private boolean isTerminalState(State state) {

		State.Turn turn = state.getTurn();

		return turn.equals(State.Turn.WHITEWIN)
				|| turn.equals(State.Turn.BLACKWIN)
				|| turn.equals(State.Turn.DRAW);
	}

	/**
	 * Crea una copia profonda dello stato
	 */
	/**
	 * Crea una copia profonda dello stato usando Gson
	 */
	private State deepCopyState(State state) {
		Gson gson = new Gson();
		String json = gson.toJson(state);

		// Determina la classe corretta (StateTablut, StateBrandub, etc.)
		State stateCopy = gson.fromJson(json, state.getClass());

		return stateCopy;
	}

	private State applyMoveCopy(State state, Action action) {
		// Crea una copia profonda
		State copy = deepCopyState(state);

		// Applica la mossa alla copia
		return movePawn(copy, action);
	}

	private State movePawn(State state, Action a) {
		State.Pawn pawn = state.getPawn(a.getRowFrom(), a.getColumnFrom());
		State.Pawn[][] newBoard = state.getBoard();

		// Libero la casella di partenza
		if (a.getColumnFrom() == 4 && a.getRowFrom() == 4) {
			newBoard[a.getRowFrom()][a.getColumnFrom()] = State.Pawn.THRONE;
		} else {
			newBoard[a.getRowFrom()][a.getColumnFrom()] = State.Pawn.EMPTY;
		}

		// Metto la pedina nella casella di arrivo
		newBoard[a.getRowTo()][a.getColumnTo()] = pawn;
		state.setBoard(newBoard);

		// Cambio il turno
		State.Turn currentTurn = state.getTurn();
		if (currentTurn.equals(State.Turn.WHITE)) {
			state.setTurn(State.Turn.BLACK);
		} else {
			state.setTurn(State.Turn.WHITE);
		}

		return state;
	}

}
