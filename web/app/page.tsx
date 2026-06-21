import Link from "next/link";
import { listPeople, thumbnailUrl } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const people = await listPeople();

  return (
    <main className="mx-auto max-w-4xl p-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Enrolled people</h1>
        <Link
          href="/enroll"
          className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
        >
          Enroll new person
        </Link>
      </div>

      {people.length === 0 ? (
        <p className="text-gray-500">No one enrolled yet.</p>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
          {people.map((person) => (
            <div key={person.id} className="rounded border p-3 text-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={thumbnailUrl(person)}
                alt={person.name}
                className="mx-auto mb-2 h-32 w-32 rounded object-cover"
              />
              <p className="font-medium">{person.name}</p>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
