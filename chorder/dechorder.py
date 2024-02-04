from .chord import Chord
from .timepoints import get_notes_by_beat
from miditoolkit.midi import containers, parser


def get_pitch_map(midi_obj):
    distribution = [0] * 12
    for instrument in midi_obj.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            distribution[note.pitch % 12] += 1

    return distribution


def get_key_from_pitch(midi_obj):
    diatonic_pitches = [0, 2, 4, 5, 7, 9, 11]
    pitch_weights = [1, 1, 1, 1, 1, 1, 1]
    distribution = get_pitch_map(midi_obj)
    max_score = -1
    max_i = -1

    for i in range(12):
        temp_distribution = distribution[i:] + distribution[:i]
        score = sum([temp_distribution[pitch] * weight for pitch, weight in zip(diatonic_pitches, pitch_weights)])

        if score > max_score:
            max_score = score
            max_i = i

    return max_i


class Dechorder:
    major_weights =      [10, -2, -1, -2, 10, -5, -2, 10, -2, -1, -2,  0]
    minor_weights =      [10, -2, -1, 10, -2, -5, -2, 10, -1, -2,  0, -2]
    diminished_weights = [10, -2, -1, 10, -2, -1, 10, -2, -2,  1, -1, -2]
    augmented_weights =  [10, -2, -1, -2, 10, -1, -2, -2, 10, -1, -2,  0]
    sus_2_weights =      [10, -2,  5, -2, -1, -5, -2,  5, -2, -1, -2,  0]
    sus_4_weights =      [10, -2, -1, -2, -5,  5, -2,  5, -2, -1, -2,  0]

    major_map =                   [0, 4, 7]
    minor_map =                   [0, 3, 7]
    diminished_map =              [0, 3, 6]
    augmented_map =               [0, 4, 8]
    dominant_map =                [0, 4, 7, 10]
    major_seventh_map =           [0, 4, 7, 11]
    minor_seventh_map =           [0, 3, 7, 10]
    diminished_seventh_map =      [0, 3, 6, 9]
    half_diminished_seventh_map = [0, 3, 6, 10]
    sus_2_map =                   [0, 2, 7]
    sus_4_map =                   [0, 5, 7]

    chord_weights = {
        'M': major_weights,
        'm': minor_weights,
        'o': diminished_weights,
        '+': augmented_weights,
        'sus2': sus_2_weights,
        'sus4': sus_4_weights
    }

    chord_maps = {
        'M': major_map,
        'm': minor_map,
        'o': diminished_map,
        '+': augmented_map,
        '7': dominant_map,
        'M7': major_seventh_map,
        'm7': minor_seventh_map,
        'o7': diminished_seventh_map,
        '/o7': half_diminished_seventh_map,
        'sus2': sus_2_map,
        'sus4': sus_4_map
    }

    @staticmethod
    def get_chord_map(notes, start=0, end=1e7):
        # Maps note to cumulative duration
        chord_map = [0] * 12
        for note in notes:
            chord_map[note.pitch % 12] += min(end, note.end) - max(start, note.start)

        return chord_map

    @staticmethod
    def get_pitch_distribution(notes, start=0, end=1e7):
        # Maps pitch to cumulative duration
        distribution = {}
        for note in notes:
            if note.pitch not in distribution.keys():
                distribution[note.pitch] = 0
            distribution[note.pitch] += min(note.end, end) - max(note.start, start)

        return distribution

    @staticmethod
    def get_bass_pc(notes, start=0, end=1e7):
        # Return the first low note that has a duration of longer than 1/8 of (start, end)
        # TODO: For 2-bar wise calculations, this threshold is thus twice as long
        distribution = Dechorder.get_pitch_distribution(notes, start, end)
        distribution = list(sorted(distribution.items()))
        for pitch, span in distribution:
            if span >= (end - start) / 8:
                return pitch % 12

        return None

    @staticmethod
    def get_chord_quality(notes, start=0, end=1e7, consider_bass=False):
        max_score = 0
        max_root = -1
        max_quality = None
        # 1. Get map from note to cumulative duration
        chord_map = Dechorder.get_chord_map(notes, start, end)
        # 2. Get bass pitch class based on duration
        bass_pc = Dechorder.get_bass_pc(notes, start, end)

        # 3. Loop through all possible roots and qualities
        for i in range(12):
            # If root is not present, skip it
            temp_map = chord_map[i:] + chord_map[:i]
            if temp_map[0] == 0:
                continue

            for quality, weights in Dechorder.chord_weights.items():
                score = sum([map_item * weight for map_item, weight in zip(temp_map, weights)])

                # TODO: does this make sense? if the pitch identified as bass is weighted < 0, then divide the score by 2
                # this only works if the bass pitch is identified correctly, meaning the bass pitch is the lowest pitch in 
                # the MIDI lasting 1/8 the interval
                if consider_bass and bass_pc is not None:
                    if weights[bass_pc] < 0:
                        score /= 2

                if score > max_score:
                    max_score = score
                    max_root = i
                    max_quality = quality

        # TODO: does this make sense? what about very sparse MIDI files? add option to relax this requirement
        if max_score < (end - start) * 10:
            return Chord(), -1

        # TODO: do these inversion rules make sense? 
        if bass_pc is not None:
            if max_quality == 'o':
                if (max_root - bass_pc + 12) % 12 == 4:
                    max_root = bass_pc
                    max_quality = '7'
                elif (max_root - bass_pc + 12) % 12 == 3:
                    max_root = bass_pc
                    max_quality = 'o7'
            elif max_quality == 'M' and (max_root - bass_pc + 12) % 12 == 3:
                max_root = bass_pc
                max_quality = 'm7'
            elif max_quality == 'm':
                if (max_root - bass_pc + 12) % 12 == 4:
                    max_root = bass_pc
                    max_quality = 'M7'
                elif (max_root - bass_pc + 12) % 12 == 3:
                    max_root = bass_pc
                    max_quality = '/o7'
            elif max_quality == 'sus2' and (max_root - bass_pc + 12) % 12 == 5:
                max_root = bass_pc
                max_quality = 'sus4'
            elif max_quality == 'sus4' and (max_root - bass_pc + 12) % 12 == 7:
                max_root = bass_pc
                max_quality = 'sus2'

        # print(max_score)

        if max_quality == 'M':
            if chord_map[(max_root + 11) % 12] > 0 and chord_map[(max_root + 11) % 12] > chord_map[(max_root + 10) % 12]:
                max_quality = 'M7'
            elif chord_map[(max_root + 10) % 12] > 0 and chord_map[(max_root + 10) % 12] > chord_map[(max_root + 11) % 12]:
                max_quality = '7'
        if max_quality == 'm' and chord_map[(max_root + 10) % 12] > 0:
            max_quality = 'm7'

        return Chord(args=(max_root, max_quality, bass_pc)), max_score

    @staticmethod
    def get_chords(midi_obj, start=None, end=None, beat=2):
        interval = midi_obj.ticks_per_beat * beat

        # for debugging
        # for i, notes in enumerate(get_notes_by_beat(midi_obj, beat, start, end)):
        #     chd = Dechorder.get_chord_quality(notes, start + i * interval, start + (i + 1) * interval)

        return [Dechorder.get_chord_quality(notes, start + i * interval, start + (i + 1) * interval) for i, notes in
                enumerate(get_notes_by_beat(midi_obj, beat, start, end))]

    @staticmethod
    def dechord(midi_obj, scale=None):
        if scale is None:
            scale = Chord.default_scale

        max_note_tick = max([max([n.end for n in i.notes]) for i in midi_obj.instruments]) + 1


        # TODO: parse depending on time signature 
        
        # if numerator divisible by 2, 2-bar parsing vs 1-bar parsing
        # if numerator divisible by 3, 3-bar parsing vs 1-bar parsing
        # otherwise, for numerator n, n-bar vs 1-bar parsing
            
        chords = []

        # TODO: default for ts not divisible by 2 or 3

        if len(midi_obj.time_signature_changes) == 0:
        # if True: # TODO
            ts_changes = [containers.TimeSignature(numerator=4, denominator=4, time=0)]
        else:
            ts_changes = midi_obj.time_signature_changes
            
        for ts_i, ts_change in enumerate(ts_changes):

            if ts_i + 1 < len(ts_changes):
                next_ts_change = ts_changes[ts_i + 1]
                next_ts_time = next_ts_change.time
            else:
                next_ts_time = max_note_tick

            ts_num = ts_change.numerator
            ts_den = ts_change.denominator
            ts_time = ts_change.time

            # TODO: ts depednent parsing
            # if ts_num % 2 == 0:
            if True: # TODO

                # Compute chords every 1 beat
                chords_1 = Dechorder.get_chords(midi_obj, beat=1, start=ts_time, end=next_ts_time)
                # Compute chords every 2 beats
                chords_2 = Dechorder.get_chords(midi_obj, beat=2, start=ts_time, end=next_ts_time)

                # Downweight 2-beat wise scores (divide by 2)
                for i in range(len(chords_2)):
                    chord = chords_2[i]
                    chords_2[i] = chord[0], chord[1] / 2

                # Iterate through every 2 bars
                for i in range(len(chords_2)):
                    prev_index = i * 2
                    next_index = i * 2 + 1
                    # Get the 2-bar wise assignment and score for this beat and the previous one
                    two_chord = chords_2[i]
                    two_score = two_chord[1]
                    # Get the 1-bar wise assignment and score for the previous beat
                    prev_chord = chords_1[prev_index]
                    prev_score = prev_chord[1]

                    # If there is a next bar, choose the 1-beat wise assignment if BOTH 1-beats wise assignments are better,
                    # otherwise choose the 2-beat wise assignment
                    if next_index < len(chords_1):
                        next_chord = chords_1[next_index]
                        next_score = next_chord[1]

                        # print(f'{prev_score} {next_score} {two_score}')
                        if prev_score >= two_score and next_score >= two_score:
                            chords += [prev_chord[0], next_chord[0]]
                        else:
                            chords += [two_chord[0]] * 2
                    
                    # If there is not a next bar, choose the 1-bar wise assignment if it is better, otherwise choose the
                    # 2-bar wise assignment
                    else:
                        if prev_score > two_score:
                            chords.append(prev_chord[0])
                        else:
                            chords.append(two_chord[0])

            elif ts_num % 3 == 0:

                chords_1 = Dechorder.get_chords(midi_obj, beat=1, start=ts_time, end=next_ts_time)
                chords_3 = Dechorder.get_chords(midi_obj, beat=3, start=ts_time, end=next_ts_time)

                for i in range(len(chords_3)):
                    chord = chords_3[i]
                    chords_3[i] = chord[0], chord[1] / 3
                
                # Iterate through every 3 bars
                for i in range(len(chords_3)):
                    index_1 = i * 2
                    index_2 = i * 2 + 1
                    index_3 = i * 2 + 2
                    # Get the 3-bar wise assignment and score for this beat and previous 2
                    three_chord = chords_3[i]
                    three_score = three_chord[1]
                    # Get the two 1-bar wise assignments and score for the previous two beat
                    one_chord = chords_1[index_1]
                    one_score = one_chord[1]

                    if index_3 < len(chords_1):
                        two_chord = chords_1[index_2]
                        two_score = two_chord[1]

                        next_chord = chords_1[index_3]
                        next_score = next_chord[1]

                        if one_score >= three_score and two_score >= three_score and next_score >= three_score:
                            chords += [one_chord[0], two_chord[0], next_chord[0]]
                        else:
                            chords += [three_chord[0]] * 3

                    elif index_2 < len(chords_1):
                        next_chord = chords_1[index_2]
                        next_score = next_chord[1]
                        
                        if one_score >= three_score and next_score >= three_score:
                            chords += [one_chord[0], next_chord[0]]
                        else:
                            chords += [three_chord[0]] * 2

                    else:
                        if one_score > three_score:
                            chords.append(one_chord[0])
                        else:
                            chords.append(three_chord[0])

        return chords

    @staticmethod
    def enchord(midi_obj):
        interval = midi_obj.ticks_per_beat
        midi_obj.markers = []
        chords = Dechorder.dechord(midi_obj)
        first_chord = chords[0]

        if first_chord.is_complete():
            chord_marker = containers.Marker(text=str(first_chord), time=0)
            midi_obj.markers.append(chord_marker)
        for i, (prev_chord, next_chord) in enumerate(zip(chords[:-1], chords[1:])):
            if prev_chord != next_chord:
                if next_chord.is_complete():
                    chord_marker = containers.Marker(text=str(next_chord), time=(i + 1) * interval)
                    midi_obj.markers.append(chord_marker)

        return midi_obj


def chord_to_midi(chord):
    if not chord.is_complete():
        return None

    root_c = 60
    bass_c = 36
    root_pc = chord.root_pc
    chord_map = Dechorder.chord_maps[chord.quality]
    bass_pc = chord.bass_pc

    return [bass_c + bass_pc] + [root_c + root_pc + i for i in chord_map]


def play_chords(midi_obj):
    default_velocity = 63
    midi_maps = [chord_to_midi(Chord(marker.text)) for marker in midi_obj.markers]
    new_midi_obj = parser.MidiFile()
    new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    # new_midi_obj.time_signature_changes.append(containers.TimeSignature(numerator=4, denominator=4, time=0))
    new_midi_obj.time_signature_changes.extend(midi_obj.time_signature_changes)
    new_midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='Piano'))

    for midi_map, prev_marker, next_marker in zip(midi_maps, midi_obj.markers[:-1], midi_obj.markers[1:]):
        for midi_pitch in midi_map:
            midi_note = containers.Note(start=prev_marker.time, end=next_marker.time, pitch=midi_pitch,
                                        velocity=default_velocity)
            new_midi_obj.instruments[0].notes.append(midi_note)

    return new_midi_obj
